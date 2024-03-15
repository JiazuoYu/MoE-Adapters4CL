##!/bin/bash
set -v
set -e
set -x
exp_no=withFrozen_22experts_3000epoch
GPU=0,1
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
dataset_test=(Aircraft Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-3 1e-3 5e-3 1e-3 1e-4 1e-3 1e-3 1e-4 1e-3 1e-3 1e-3)
chooser=(TinyImagenet_autochooser Aircraft_autochooser Caltech101_autochooser CIFAR100_autochooser DTD_autochooser EuroSAT_autochooser Flowers_autochooser Food_autochooser MNIST_autochooser OxfordPet_autochooser StanfordCars_autochooser SUN397_autochooser)
threshold=(655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4 655e-4)  # 手动设置
num=22 # experts num
frozen_path=frozen_list_experts_3000epoch

# test model_ckpt_path
model_ckpt_path=ckpt/exp_${exp_no}

# train
j=0
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=adapter \
    --train-dataset=${dataset[j]} \
    --lr=${lr[j]} \
    --ls 0.2 \
    --iterations 3000 \
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
        --iterations 3000 \
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
for ((j = 1; j < 12; j++)); do
  for ((i = 1; i < ${#dataset_test[@]}; i++)); do
    dataset_cur=${dataset_test[j]}

    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval-only \
        --train-mode=adapter \
        --eval-datasets=${dataset_cur} \
        --load ${model_ckpt_path}/${dataset_test[i]}.pth \
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
