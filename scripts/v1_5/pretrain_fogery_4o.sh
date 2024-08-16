#!/bin/bash
export MASTER_PORT=34229
GPUS=${GPUS:-1}
##写一个一加一的脚本
# deepspeed llava/train/train_mem.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --mm_projector_lr 1e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/data1/zhangzr22/LLaVA_DATA/llava-v1.5-7b \
    --version v1 \
    --data_path /home/data1/zhangzr22/LLaVA_DATA/0717/qa_pairs_caption_v2.json \
    --image_folder /home/data1/zhangzr22/LLaVA_DATA/sft_forgery_data \
    --vision_tower /home/data1/zhangzr22/clip/clip-vit-large-patch14 \
    --ctx_path /home/data1/zhangzr22/LLaVA_DATA/0717/ctx_token.bin \
    --train_pixel_model True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --image_aspect_ratio no \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-pretrain-forgery-v1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 7000 \
    --train_pixel_model False \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard 
    # --fp16 True
