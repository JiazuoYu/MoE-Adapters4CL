##!/bin/bash

set -v
set -e
set -x
GPU=3,4
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-3 1e-3 5e-3 1e-3 1e-4 1e-3 1e-3 1e-4 1e-3 1e-3 1e-3)
chooser=(Aircraft_autochooser Caltech101_autochooser CIFAR100_autochooser DTD_autochooser EuroSAT_autochooser Flowers_autochooser Food_autochooser MNIST_autochooser OxfordPet_autochooser StanfordCars_autochooser SUN397_autochooser)
threshold=(655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4)
num=22 # experts num

###  only need to set your ckpt_path ###
model_ckpt_path=ckpt/full_shot_order1_1000iters

# inference
for ((j = 0; j < 11; j++)); do
  for ((i = 0; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[j]}

    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval-only \
        --train-mode=adapter \
        --eval-datasets=${dataset_cur} \
        --load ${model_ckpt_path}/${dataset[i]}.pth \
        --load_autochooser ${model_ckpt_path}/${chooser[i]}.pth \
        --data-location /home/dhw/yjz_workspace/data/data \
        --ffn_adapt_where AdapterDoubleEncoder \
        --ffn_adapt \
        --apply_moe \
        --task_id 200 \
        --multi_experts \
        --experts_num ${num} \
        --autorouter \
        --threshold=${threshold[i]}
    done
done
