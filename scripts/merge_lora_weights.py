import argparse
import sys
sys.path.append('/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
import torch.nn as nn

# Assuming `model` is already defined and instantiated

def check_model_weights_for_nan(model: nn.Module):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN values found in parameter: {name}")
        else:
            # print(f"No NaN values in parameter: {name}")
            pass

def merge_lora(args):
    # breakpoint()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')
    check_model_weights_for_nan(model)
    breakpoint()
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent/checkpoints/llava-v1.5-7b-fogery-lora-0804-gt-yesno-sft-Casiav2_v1')
    parser.add_argument("--model-base", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-onlytest-yesno-sft-7b')
    parser.add_argument("--save-model-path", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-onlytest-yesno-sft-Casiav2-7b')
    # parser.add_argument("--model-path", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/0717/LLaVA_agent/checkpoints/llava-pretrain-finetune-forgery-r16-lora')
    # parser.add_argument("--model-base", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-v1.5-7b')
    # parser.add_argument("--save-model-path", type=str, default='/home/data1/zhangzr22/LLaVA_DATA/llava-forgery-test-7b')
    args = parser.parse_args()

    merge_lora(args)
