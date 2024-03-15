#!bin/bash

# for imagenet-100 dataset; 10 classes/task
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"

