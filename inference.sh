#!/usr/bin/env bash

python inference.py \
--random_seed 42 \
--sam_checkpoint "<path-to-vanilla-sam-checkpoint>" \
--checkpoint "<path-to-custom-checkpoint>" \
--model_type "vit_b" \
--device "cpu" \
--image_size 256 \
--data_root "<path-to-dataset-root>" \
--batch_size 8 \
--num_workers 8 \
--metrics "iou" "dice" "precision" "f1_score" "recall" "specificity" "accuracy" \
--point_num 1 \
--boxes_prompt \
--iter_point 8 \
--multimask \
--pred_iou_thresh 0.88 \
--stability_score_thresh 0.95 \
--points_per_side 32 \
--points_per_batch 256