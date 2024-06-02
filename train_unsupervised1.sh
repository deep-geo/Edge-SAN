#!/usr/bin/env bash

python train.py \
--work_dir "/root/autodl-tmp/workdir" \
--run_name "nuclei" \
--seed 42 \
--epochs 100000 \
--batch_size 16 \
--num_workers 16 \
--image_size 256 \
--mask_num 5 \
--split_paths  "/root/autodl-tmp/datasets/ALL2/split.json" \
#--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'specificity' 'accuracy' \
--metrics 'iou' 'dice' 'precision' 'f1_score' 'recall' 'aji' 'dq' 'sq' 'pq'\
--device "cuda" \
--lr 0.0001 \
--resume "/root/NucleiSAM/pretrain_model/epoch0058_test-loss0.1156_sam.pth" \
--model_type "vit_b" \
--boxes_prompt \
--point_num 1 \
--iter_point 8 \
--multimask \
--encoder_adapter \
--activate_unsupervised \
--unsupervised_dir "/root/autodl-tmp/datasets/unsupervised/cell_images/" \
--unsupervised_start_epoch 0 \
--unsupervised_step 1 \
#--prompt_path
#--save_pred
#--lr_scheduler \
#--point_list
