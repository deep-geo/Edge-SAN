#!/usr/bin/env bash

python train.py \
--work_dir "workdir" \
--run_name "<run_name>" \
--epochs 100000 \
--batch_size 8 \
--num_workers 8 \
--image_size 256 \
--mask_num 5 \
--split_paths \
  "<split_path_0>" \
  "<split_path_1>" \
  "<split_path_2>" \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'hausdorff_distance' \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--multimask \
--encoder_adapter \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
