import copy
import os
from random import random
import numpy as np
import clip
import torch
import csv
import pandas as pd
from . import utils
from .args import parse_arguments
from .models import evaluate, evaluate_wise_ft, finetune, finetune_icarl, Autoencoder, Alexnet_FE, few_shot_AutoEncoder, AutoEncoder, few_shot_autoencoder
from .models.modeling import create_image_classifier
import torchvision.models as models
import torch.nn as nn

# def merge(model_0, model_1, alpha=0.95):
#     key_name = [k for k, v in model_0.named_parameters()]
#     for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
#         param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
#     return model_1


def write2csv(i,map_v,map_t, existing_data):


    top_values_v, top_indices_v = torch.topk(map_v, 2)
    top_values_t, top_indices_t = torch.topk(map_t, 2)
    map_v_combine = torch.cat((map_v, top_indices_v), dim=0)
    map_t_combine = torch.cat((map_t, top_indices_t), dim=0)
    map_v_combine = np.array(map_v_combine)
    map_t_combine = np.array(map_t_combine)
    existing_data.iloc[i+12] = map_v_combine
    existing_data.iloc[i] = map_t_combine
    return existing_data
    # merged_data = pd.concat([existing_data, new_row], ignore_index=True)
    # existing_data = existing_data.append(new_row, ignore_index=True)


def main(args):
    utils.seed_all(args.seed)

    assert args.train_mode in ["whole", "text", "image", "adapter"]
    if args.eval_only:  # 测试阶段
        model, _, val_preprocess = clip.load(args.model, jit=False, args=args)
        if args.load:    #
            utils.torch_load(model, args.load)
        if args.load_autochooser and args.autorouter == True:
            pretrained_alexnet = models.alexnet(pretrained=True)
            feature_extractor = Alexnet_FE(pretrained_alexnet).cuda()
            Autoencoder_list = nn.ModuleList()
            for i in range(args.task_num+1):  # more for zero-shot chosen  / few or full shot share the code
                model_autoencoder = Autoencoder(256 * 13 * 13)
                Autoencoder_list.append(model_autoencoder)
            utils.torch_load(Autoencoder_list, args.load_autochooser)
            Autoencoder_list = Autoencoder_list.cuda()
        elif args.save:  # None
            checkpoint_pth = os.path.join(
                args.save, f"clip_zeroshot_{args.train_dataset}.pth"
            )
            utils.torch_save(checkpoint_pth, model)
        evaluate(model,feature_extractor,Autoencoder_list, args, val_preprocess)


    else:
        if args.train_chooser:
            if args.few_shot > 0:
                print('----------------------train few-shot chooser----------------------')
                chooser_of_few_shot = few_shot_AutoEncoder(args)# few shot chooser
            else:
                print('----------------------train full-shot chooser----------------------')
                chooser = AutoEncoder(args)
        else:
            print('----------------------finetune model----------------------')
            model = finetune(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
