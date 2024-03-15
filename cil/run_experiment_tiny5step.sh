#!bin/bash

# for imagenet-100 dataset; 10 classes/task
python main.py \
    --config-path configs/class \
    --config-name tinyimagenet_100-20.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/tinyimagenet.yaml"

# for imagenet-1000 dataset; 100 classes/task
# python main.py \
#     --config-path configs/class \
#     --config-name imagenet1000_100-100.yaml \
#     dataset_root="../datasets/" \
#     class_order="class_orders/imagenet1000.yaml"