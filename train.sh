#!/usr/bin/env bash

python train.py \
--work_dir "/root/autodl-tmp/workdir" \
--run_name "supervised baseline" \
--seed 42 \
--epochs 100 \
--batch_size 8 \
--test_sample_rate 0.2 \
--num_workers 8 \
--image_size 256 \
--mask_num 5 \
--split_paths "/root/autodl-tmp/ALL3/split.json" \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'aji' 'dq' 'sq' 'pq' \
--sam_checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "/root/autodl-tmp/epoch0026_test-loss0.1573_sam.pth" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--encoder_adapter \
--multimask \
--pred_iou_thresh 0.80 \
--stability_score_thresh 0.90 \
--points_per_side 32 \
--points_per_batch 256 \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list

#"/root/autodl-fs/workdir/models/supervised baseline_05-31_08-59/epoch0001_test-loss0.2279_sam.pth" \
#--sam_checkpoint "autodl-tmp/sam-med2d_b.pth" \
