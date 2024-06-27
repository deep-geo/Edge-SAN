#!/usr/bin/env bash

python train.py \
--work_dir "workdir" \
--run_name "nuclei" \
--seed 42 \
--epochs 100000 \
--batch_size 8 \
--num_workers 8 \
--eval_interval 100 \
--test_sample_rate 1.0 \
--image_size 256 \
--mask_num 5 \
--data_root "/root/autodl-tmp/datasets/SAM_nuclei_preprocessed/ALL_Multi" \
--test_size 0.1 \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' 'aji' 'dq' 'sq' 'pq' \
--checkpoint "/root/autodl-tmp/sam_vit_b_01ec64.pth" \
--device "cuda" \
--lr 0.0001 \
--resume "" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--edge_point_num 3 \
--iter_point 8 \
--encoder_adapter \
--multimask \
--activate_unsupervised \
--unsupervised_dir "/root/autodl-tmp/datasets/SAM_nuclei/<unsupervised_root>" \
--unsupervised_start_epoch 1 \
--unsupervised_step 1 \
--unsupervised_weight 1.0 \
--unsupervised_weight_gr 0.0 \
--unsupervised_initial_sample_rate 0.1 \
--unsupervised_sample_rate_delta 0.05 \
--unsupervised_metric_delta_threshold 0.01 \
--unsupervised_focused_metric "Overall/dice" \
--pred_iou_thresh 0.88 \
--stability_score_thresh 0.95 \
--points_per_side 32 \
--points_per_batch 256 \
#--unsupervised_only \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
