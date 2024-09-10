# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/data1/zhangzr22/LLaVA_DATA/mae_raw')
import open_clip
from prompt_ensemble import AnomalyCLIP_PromptLearner
from collections import OrderedDict

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]   #768
        num_patches = model.patch_embed.num_patches   #32*32=1024
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches #1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5) #14
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5) #32
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]  #1, 1, 768
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] #1, 196, 768
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)  # 1, 768, 14, 14
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False) # 1, 768, 32, 32
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2) #1,1024,768
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x
    

class noise_vit_tiny(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(noise_vit_tiny, self).__init__(**kwargs)

        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.load_weight()

    def forward(self, x):
        output = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        x_ = x[:, 1:, :]
        
        output.append(self.norm(x_))
        for i, blk in enumerate(self.blocks, start=0):
            if (i + 1) % 4 ==0:
                x = blk(x)
                x_ = x[:, 1:, :]
                output.append(self.norm(x_))
            else:
                x = blk(x)

        return output
    
    def load_weight(self):
        state_dict = torch.load(r'/home/data1/zhangzr22/zfr/mae_raw/mae_tiny_400e.pth.tar')
        state_dict = state_dict['model']
        new_state_dict = OrderedDict()
        for name, params in state_dict.items():
            new_state_dict[name[13:]] = state_dict[name]
        interpolate_pos_embed(self, new_state_dict)
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(msg)
        

class ScaledDot_txt(nn.Module):
    def __init__(self, mask_token=441, mask_value=-1e9):
        super(ScaledDot_txt, self).__init__()
        self.mask_value = mask_value
        self.Q = nn.Parameter(torch.randn(1, mask_token, 2))  # 改为 441 以确保输出维度为 441
        # self.linear = nn.Linear(mask_token, 441)  # 添加线性变换来调整输出维度

    def forward(self, mask, attn_mask=None):
        '''
        Q: [batch_size, 441, d_k]
        K: [batch_size, len_k, d_k]
        V: [batch_size, len_v(=len_k), d_v]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B = mask.size(0)
        Q = self.Q.repeat(B, 1, 1)  # 扩展 Q 的形状为 (batch_size, 441, 768)
        d_k = Q.size(-1)

        # 计算注意力得分
        scores = torch.matmul(Q, mask.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, 441, token]

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, self.mask_value)

        # 计算注意力权重
        attn = nn.Softmax(dim=-1)(scores)  # attn : [batch_size, 441, token]

        # # 线性变换将 attn 维度调整到 441
        # attn = self.linear(attn.transpose(1, 2)).transpose(1, 2)  # attn : [batch_size, 441, token]

        # 计算上下文向量
        context = torch.matmul(attn, mask)  # [batch_size, 441, 768]

        return context  # 输出维度为 [batch_size, 441, 特征维度]
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, emb_dim=768, mask_value=-1e9):
        super(ScaledDotProductAttention, self).__init__()
        self.mask_value = mask_value

        # self.Wq = nn.Linear(emb_dim, emb_dim)
        # self.Wk = nn.Linear(emb_dim, emb_dim)
        # self.Wv = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, Q, attn_mask=None):
        '''
        Q: [batch_size, len_q, d_k]
        K: [batch_size, len_k, d_k]
        V: [batch_size, len_v(=len_k), d_v]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # breakpoint()
        # Q = self.Wq(Q)
        # K = self.Wk(x)
        # V = self.Wv(x)
        K = x
        V = x
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, len_q, len_k]
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, self.mask_value)
 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, len_q, d_v]
        return context


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])     # (1,1370,1024) => (1,1369,768)
            else:
                B, C, _, _ = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
    

class Noisefusion(nn.Module):
    def __init__(self, emb_dim, k):
        super().__init__()
        """ k: length of feature_list
        """
        self.fuse = nn.ModuleList([ScaledDotProductAttention(emb_dim=emb_dim) for i in range(k)])

    def forward(self, image_features, noise_features):
        """ image_features: list
            noise_feature: [B, L, Embeding]
        """
        fusion = []
        for i, image_feature in enumerate(image_features):
            noise_feature = noise_features[i]
            tmp_fusion = self.fuse[i](noise_feature, image_feature) # Q: image feature, K\V: noise_feature
            fusion.append(tmp_fusion)

        return torch.stack(fusion, dim=0).mean(dim=0)
    

class TextSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features):
        """ image_features: list
            txt_embedding: [B, embedding_length, 2]
        """
        similarity_list = []
        # image_features_modified = self.linear(image_features)
        for i, image_feature in enumerate(image_features):
            # 1. norm
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            # 2. similarity, (1,1024,768)@(1,768,2) => (1,1024,2)
            similarity = image_feature @ text_features.permute(0, 2, 1)
            similarity_list.append(similarity)


        return torch.stack(similarity_list, dim=0).mean(dim=0)

class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(features[-1], out_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        return x
    


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        embed_dim = kwargs['embed_dim']
        self.img_size = kwargs['img_size']
        self.patch_size = kwargs['patch_size']
        self.feature_list = [6, 12, 18, 24]
        # adjust feature list
        self.trainable_linearlayer = LinearLayer(1024, embed_dim, 4)
        self.per_num_tokens = self.img_size // self.patch_size
        # num_tokens = self.per_num_tokens**2
        # self.proj_img = nn.Linear(num_tokens+1, num_tokens)
        
        # noise
        # self.constrain_conv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
        # self.noise_vit = noise_vit_tiny(img_size=self.img_size, patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.noise_fusion = Noisefusion(emb_dim=embed_dim, k=4)
        self.proj = nn.Linear(embed_dim, 2)

        # clip
        self.clipmodel, _, _ = open_clip.create_model_and_transforms("/home/data1/zhangzr22/LLaVA_DATA/mae_raw_ckpt/ViT-L-14-336px.pt", self.img_size, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer("/home/data1/zhangzr22/LLaVA_DATA/mae_raw_ckpt/ViT-L-14-336px.pt")
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.clipmodel.to("cpu"), self.tokenizer)
        # 冻结模型的所有参数
        for param in self.clipmodel.parameters():
            param.requires_grad = False
        # 确认模型是否已冻结
        print("Clip model frozen:", all(not p.requires_grad for p in self.clipmodel.parameters()))
        # self.tokenizer = open_clip.get_tokenizer("/home/data1/zhangzr22/zfr/VAND-APRIL-LLM-Agent/ViT-L-14-336px.pt")
        # self.prompt_learner = AnomalyCLIP_PromptLearner(self.clipmodel.to("cpu"), self.tokenizer)
  
        self.text_fusion = TextSimilarity()
        
        self.decoder = Decoder2D(embed_dim+4, kwargs['num_classes'])
        
        self.bceloss = nn.BCEWithLogitsLoss()

        ### 降低文本特征token
        self.dot_attention = ScaledDot_txt(mask_token=441)
        

    def forward_txt_features(self):
        self.clipmodel.eval()
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(cls_id = None)
        text_features = self.clipmodel.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        # print('text_features.shape:', text_features.shape)
        text_features = torch.mean(text_features, dim=0, keepdim=True)
        return text_features
        

    def forward_features(self, x):
        feature_list = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        feature_list.append(self.norm(x[:, 1:, :]))

        for i, blk in enumerate(self.blocks, start=0):
            if (i + 1) % 4 ==0:
                x = blk(x)
                feature_list.append(self.norm(x[:, 1:, :]))
            else:
                x = blk(x)
        # x = self.proj_img(x.permute(0,2,1)).permute(0,2,1)
        x = self.norm(x)
        x_ = x[:, 1:, :]
        # h, w = self.img_size // self.patch_size, self.img_size // self.patch_size
        # outcome = rearrange(x_, 'b (h w) c -> b c h w', h=h, w=w)

        return x_, feature_list
    
    def forward_loss(self, probs, targets):
        num = targets.size(0)
        smooth = 1e-8

        # probs = torch.sigmoid(logits)
        intersection = (probs * targets)
        score = (2*torch.sum(intersection,dim=(2,3)) + smooth)/(torch.sum(probs*probs, dim=(2,3))+torch.sum(targets*targets, dim=(2,3)) + smooth)
        loss = 1 - score.sum()/num

        return loss

    def forward(self, x, gt):
        text_feature = self.forward_txt_features().bfloat16()
        input_ = x.clone()
        # input_gray = rgb2gray(input_)
        # noise = self.constrain_conv(input_gray)
        # noise_features = self.noise_vit(noise)
        with torch.no_grad():
            noise_image_feature, noise_image_feature_token, noise_feature_list = self.clipmodel.encode_image(input_, self.feature_list)
        image_feature, feature_list = self.forward_features(x)
        noise_feature_list_modified = self.trainable_linearlayer(noise_feature_list)
        noise_fusion = self.noise_fusion(feature_list, noise_feature_list_modified)
        noise_fusion = self.proj(noise_fusion)
        txt_fusion = self.text_fusion(noise_feature_list, text_feature)
        txt_fusion = self.dot_attention(txt_fusion)
        concat_feature = torch.cat((noise_fusion, image_feature, txt_fusion), dim=-1)  # [1,1024,772]
        concat_feature_ = rearrange(concat_feature, 'b (h w) c -> b c h w', h=self.per_num_tokens, w=self.per_num_tokens)
        out = self.decoder(concat_feature_)
        pred_mask = torch.sigmoid(out)

        seg_loss = self.forward_loss(pred_mask, gt)
        # bce_loss = self.bceloss(out, gt)
        # loss = 0.8*seg_loss+0.2*bce_loss
        loss = seg_loss
       
        return pred_mask, loss


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    x = torch.randn([2, 3, 336, 336]).bfloat16()
    gt = torch.randn([2, 1, 336, 336]).bfloat16()
    edge = torch.randn([2, 1, 336, 336]).bfloat16()
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), img_size=336, num_classes=1).bfloat16()
    
    y, loss = model(x, gt)
    print(y.shape)




    # k = torch.randn(2, 1024, 768)
    # Q = torch.randn(2, 1024, 768)
    # cross_atten = ScaledDotProductAttention()
    # out = cross_atten(k, Q)
    # print(out.shape)