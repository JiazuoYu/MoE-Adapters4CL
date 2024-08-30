##!/bin/bash

set -v
set -e
set -x
# 1.frozen_path 2. exp_no
exp_no=withFrozen_22experts_1000epoch_11
GPU=1,2
chooser_dataset=(TinyImagenet Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-3 1e-3 5e-3 1e-3 1e-4 1e-3 1e-3 1e-4 1e-3 1e-3 1e-3)
chooser=(Aircraft_autochooser Caltech101_autochooser CIFAR100_autochooser DTD_autochooser EuroSAT_autochooser Flowers_autochooser Food_autochooser MNIST_autochooser OxfordPet_autochooser StanfordCars_autochooser SUN397_autochooser)
threshold=(655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4)  
num=22 # experts num
frozen_path=frozen_list_experts_1000epoch

# test model_ckpt_path
model_ckpt_path=ckpt/exp_${exp_no}

# train

# train chooser of DDAS
j=0
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=adapter \
    --train-dataset=${chooser_dataset[j]} \
    --iterations 1000 \
    --method finetune \
    --save ckpt/exp_${exp_no} \
    --data-location /home/dhw/yjz_workspace/data/data \
    --task_id ${j} \
    --is_train \
    --train_chooser
for ((i = 1; i < ${#chooser_dataset[@]}; i++)); do
    dataset_cur=${chooser_dataset[i]}
    dataset_pre=${chooser_dataset[i - 1]}
    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
        --train-mode=adapter \
        --train-dataset=${dataset_cur} \
        --ls 0.2 \
        --method finetune \
        --iterations 300 \
        --save ckpt/exp_${exp_no} \
        --load ckpt/exp_${exp_no}/${dataset_pre}_autochooser.pth \
        --data-location /home/dhw/yjz_workspace/data/data \
        --is_train \
        --task_id ${i} \
        --train_chooser
done

# train MoE-Adapters
j=0
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=adapter \
    --train-dataset=${dataset[j]} \
    --lr=${lr[j]} \
    --ls 0.2 \
    --iterations 1000 \
    --method finetune \
    --save ckpt/exp_${exp_no} \
    --data-location /home/dhw/yjz_workspace/data/data \
    --ffn_adapt_where AdapterDoubleEncoder\
    --ffn_adapt \
    --task_id ${j} \
    --multi_experts \
    --apply_moe \
    --frozen-path ${frozen_path}${num} \
    --experts_num ${num} \
    --is_train


for ((i = 1; i < ${#dataset[@]}; i++)); do
#for ((i = 2; i < 10; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
        --train-mode=adapter \
        --train-dataset=${dataset_cur} \
        --lr=${lr[i]} \
        --ls 0.2 \
        --method finetune \
        --iterations 1000 \
        --save ckpt/exp_${exp_no} \
        --load ckpt/exp_${exp_no}/${dataset_pre}.pth \
        --data-location /home/dhw/yjz_workspace/data/data \
        --ffn_adapt_where AdapterDoubleEncoder \
        --ffn_adapt \
        --apply_moe \
        --repeat_train \
        --multi_experts \
        --frozen \
        --frozen-path ${frozen_path}${num} \
        --experts_num ${num} \
        --is_train \
        --task_id ${i}
done


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
