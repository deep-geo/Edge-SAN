#!/usr/bin/env bash

python inference.py \
--sam_checkpoint "/root/autodl-tmp/workdir/models/supervised baseline All3_05-29_13-39/epoch0002_test-loss0.2714_sam.pth" \
--model_type "vit_h" \
--device "cuda" \
--image_size 256 \
--data_root "/root/autodl-tmp/MoNuSeg2020/" \
--batch_size 16 \
--num_workers 8 \
--metrics "iou" "dice" "f1_score" \
--point_num 1 \
--boxes_prompt \
--iter_point 8 \
--multimask