#!/usr/bin/env bash

python test.py \
--run_name "test" \
--seed 42 \
--batch_size 8 \
--num_workers 8 \
--image_size 256 \
--data_root "/root/autodl-tmp/MoNuSeg2020" \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--model_type "vit_b" \
--device "cuda" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--encoder_adapter \
--multimask

#"/root/autodl-fs/workdir/models/supervised baseline_05-31_08-59/epoch0001_test-loss0.2279_sam.pth" \
#--sam_checkpoint "autodl-tmp/sam-med2d_b.pth" \
