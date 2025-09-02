##!/bin/bash

set -v
set -e
set -x

: <<'EOF'
The test script of MoE-Adapters++ for full-shot under order1

to save and check the result, run:
bash train_full_shot_1000iters_order1.sh -> result.txt
you can get all logs in "result.txt" 

in infreence, we only need to set:
    1. the name of log file
    2. the name & path of checkpoints and data
    3. learning rate
    4. GPU
    5. layer settings

in inference, we only need to set:
    1. the name of log file
    2. the path of checkpoints and data
    3. zero-shot threshold for LEAS 
    4. GPU
    5. ensure layers are set up and checkpoints are consistent
EOF

# TODO: 1. set the name of the checkpoint and log file
date=0919
exp_setting=base

# TODO: 2. set the path of checkpoints and data
exp_no=${date}_${exp_setting}
# test model_ckpt_path
model_ckpt_path=ckpt/Order1/full_shot/
tensorboard_log_path=logs/Order1/full_shot/tensorboard/${exp_no}
data_location=/home/dhw/yjz_workspace/data/data

# TODO: 4. Currently only supports single GPU
GPU=2

# TODO: Dataset & LR
dataset=(StanfordCars Food MNIST OxfordPet Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
lr_ce=(1e-2 1e-4 1e-4 1e-4 2e-3 1e-3 1e-4 5e-3 5e-3 8e-2 5e-5)
lr_ae=(1e-3 2e-3 2e-4 5e-3 2e-3 5e-3 1e-2 2e-3 1e-4 1e-3 5e-3)

: <<'EOF'
setting for Dynamic MoE-Adapters

list_v: 
    layers of image encoder, 
    first layer of "1" is the recognition layer, 
    other layers are subsequent layers

list_t is similar to list_v

list_noly_LEAS:
    "mutil-LEAS" setting in paper,
    only deploy LEAS in shallow layers before the recognition layer

list_v_force:
    force training strategy in paper
    in subsequent tasks, expansion a new expert and AE in current layer at first iteration 
    The default option is to apply force training strategy at the recognition layer

list_t_force is similar to list_v_force
EOF
# TODO: nsure layers are set up and checkpoints are consistent
list_v=(0 0 0 0 0 0 1 1 1 1 1 1)  # layers of image encoder, first layer of "1" is the recognition layer
list_noly_LEAS=(0 0 0 1 1 1 0 0 0 0 0 0)  # mutil-LEAS
list_t=(0 0 0 0 0 0 0 0 0 0 0 0)  # deploy in text encoder is not very well
list_v_force=(0 0 0 0 0 0 1 0 0 0 0 0)

# TODO: zero-shot threshold for LEAS
threshold=(20e-1 20e-1 20e-1 23e-1 23e-1 23e-1 23e-1 23e-1 30e-1 31e-1 31e-1) # CON

# Weighted vector of the discrepancy function 
# including "CON", "CON_norm", "STD", "STD_norm"
# in paper, "CON" is the best
discrepancy_weighted_vector=CON

# expansion threshold for DEeC
exp_threshold_image=1.0
exp_threshold_text=0.5  # if deploy MoE-Adapters++ in text encoder

# train MoE-Adapters++ init
j=0
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=adapter \
    --train-dataset=${dataset[j]} \
    --lr=${lr_ce[j]} \
    --lr_ae=${lr_ae[j]} \
    --ls 0.2 \
    --iterations 500 \
    --few_shot=5 \
    --method finetune \
    --save ${model_ckpt_path} \
    --data-location ${data_location} \
    --ffn_adapt_where AdapterDoubleEncoder\
    --ffn_adapt \
    --task_id ${j} \
    --multi_experts \
    --apply_moe \
    --is_train \
    --dyn_moe \
    --use_gate_noise \
    --expansion_threshold_image=${exp_threshold_image} \
    --expansion_threshold_text=${exp_threshold_text} \
    --log_dir ${tensorboard_log_path}/continual_${j} \
    --use_LEAS_to_eval \
    --visual_AE_hidden_dims=32 \
    --zero_shot_threshold_image=${threshold[j]} \
    --use_dyn_moe_layer_list_visual "${list_v[@]}" \
    --use_LEAS_list_visual "${list_noly_LEAS[@]}" \
    --log_dir ${tensorboard_log_path}/continual_${j} \
    --force_expansion_list "${list_v_force[@]}" \
    --discrepancy_weighted_vector=${discrepancy_weighted_vector}

# train MoE-Adapters++ continual
for ((i = 1; i < ${#dataset[@]}; i++)); do
#for ((i = 2; i < 10; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
        --train-mode=adapter \
        --train-dataset=${dataset_cur} \
        --lr=${lr_ce[i]} \
        --lr_ae=${lr_ae[j]} \
        --ls 0.2 \
        --method finetune \
        --iterations 500 \
        --few_shot=5 \
        --save ${model_ckpt_path} \
        --load ${model_ckpt_path}/${dataset_pre}.pth \
        --data-location ${data_location} \
        --ffn_adapt_where AdapterDoubleEncoder \
        --ffn_adapt \
        --apply_moe \
        --repeat_train \
        --multi_experts \
        --is_train \
        --dyn_moe \
        --use_gate_noise \
        --expansion_threshold_image=${exp_threshold_image} \
        --expansion_threshold_text=${exp_threshold_text} \
        --task_id ${i} \
        --log_dir ${tensorboard_log_path}/continual_${j} \
        --use_LEAS_to_eval \
        --visual_AE_hidden_dims=32 \
        --zero_shot_threshold_image=${threshold[j]} \
        --use_dyn_moe_layer_list_visual "${list_v[@]}" \
        --use_LEAS_list_visual "${list_noly_LEAS[@]}" \
        --log_dir ${tensorboard_log_path}/continual_${j} \
        --discrepancy_weighted_vector=${discrepancy_weighted_vector} \
        --force_expansion_list "${list_v_force[@]}"
done


# inference
for ((j = 0; j < 11; j++)); do
  for ((i = 0; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[j]}

    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval-only \
        --train-mode=adapter \
        --eval-datasets=${dataset[i]} \
        --load ${model_ckpt_path}/${dataset_cur}.pth \
        --data-location ${data_location} \
        --ffn_adapt_where AdapterDoubleEncoder \
        --ffn_adapt \
        --apply_moe \
        --task_id 200 \
        --multi_experts \
        --dyn_moe \
        --autorouter \
        --expansion_threshold_image=${exp_threshold_image} \
        --expansion_threshold_text=${exp_threshold_text} \
        --lr=${lr_ce[j]} \
        --lr_ae=${lr_ae[j]} \
        --eval_acc_task_id=${i} \
        --use_LEAS_to_eval \
        --visual_AE_hidden_dims=32 \
        --zero_shot_threshold_image=${threshold[j]} \
        --use_dyn_moe_layer_list_visual "${list_v[@]}" \
        --use_LEAS_list_visual "${list_noly_LEAS[@]}" \
        --log_dir ${tensorboard_log_path}/continual_${j} \
        --discrepancy_weighted_vector=${discrepancy_weighted_vector}
    done
done