#!/usr/bin/env bash

python inference.py \
--sam_checkpoint "<path-to-model-checkpoint>" \
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
--multimask