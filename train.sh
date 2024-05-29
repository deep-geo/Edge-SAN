#!/usr/bin/env bash

python train.py \
--work_dir "workdir" \
--run_name "nuclei" \
--seed 42 \
--epochs 100 \
--batch_size 8 \
--num_workers 8 \
--image_size 256 \
--mask_num 8 \
--split_paths "/root/autodl-tmp/ALL3/split.json" \
--metrics 'iou' 'dice' 'f1_score'  \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_h" \
--sam_checkpoint "/root/autodl-tmp/sam_vit_h_4b8939.pth" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--encoder_adapter \
--multimask \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
