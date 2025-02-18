#!/bin/bash
DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name VPT \
--img_aug \
--batch_size 128 \
--lr 0.001 \
--dataset_name $DATASET_NAME \
--loss_names 'TAL+memory+instance_memory' \
--optimizer Adam \
--root_dir /data/hyj/dataset \
--num_context 4 \
--prompt_depth 12 \
--num_epoch 60 \
--lora_alpha 4 \
--finetune /data/hyj/MLLM4Text/logs/Testing/pretrain-weights/best2.pth \
--temp 0.03 \
--temp_instance 0.015 \
--momentum 0.1 \
--instance_momentum 0.1 \
--margin 0.2 \
--sampler random \
--num_instance 2 \
--lora_lr 0.0001 \
--lamda1 1 \
--lamda2 0.1 

