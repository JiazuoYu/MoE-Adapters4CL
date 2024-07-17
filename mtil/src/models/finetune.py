import copy
import os
import numpy as np
import clip.clip as clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation

def finetune(args):
    # print('---1---',args.frozen_path)
    frozen_path = args.frozen_path
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'
    #  train_preprocess is_train=True val_preprocess is_train=False
    if args.load is not None and args.repeat_train is True:
        print('[fish] 11Router_22autoAdapter 叠加训练')
        utils.torch_load(model, args.load)
    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    # prepare template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # print(dataset)
    # number of iterations

    if args.few_shot > 0:
        print('=====few-shot======')
        # few-shot
        few_shot_data = {} # create few_shot data

        for images, labels in dataset.train_loader:
            for image, label in zip(images, labels):
                label = label.item()
                if label not in few_shot_data:
                    few_shot_data[label] = []
                if len(few_shot_data[label]) < args.few_shot:
                    few_shot_data[label].append(image)

        # create data_iter
        few_shot_images = []
        few_shot_labels = []

        for label, images in few_shot_data.items():
            few_shot_images.extend(images)
            few_shot_labels.extend([label] * len(images))

        few_shot_images = torch.stack(few_shot_images)
        few_shot_labels = torch.tensor(few_shot_labels)

        few_shot_dataset = torch.utils.data.TensorDataset(few_shot_images, few_shot_labels)
        few_shot_data_loader = DataLoader(few_shot_dataset, batch_size=args.batch_size, shuffle=True)
        print(len(few_shot_data_loader))

    if args.few_shot > 0:
        num_batches = len(few_shot_data_loader)
    else:  # full_shot
        num_batches = len(dataset.train_loader)
    if args.epochs is not None: # # False
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations  # 1000
    if args.eval_every_epoch:  # False
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval # none
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params
    if args.train_mode == "adapter":  # only train adapter
        print("[Training mode] Moe-Adapters")
        for k, v in model.named_parameters():  # forzen params
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False

        if args.frozen:  # frozen-strategy
            print('-------frozen--------')
            with open(frozen_path, "r") as file:
                lines = file.read().splitlines()
                frozen_list = list(set(lines))
            params = []
            params_name = []
            for k, v in model.named_parameters():
                if k in frozen_list:
                    v.requires_grad = False
                    continue
                if "adaptmlp" in k or "router" in k or "noise" in k:
                    params.append(v)
                    params_name.append(k)

            # print('frozen mode========trainable params============', params_name)
            # print('frozen mode========frozen params of trainable params============', frozen_list)
        else:
            params = [
                v for k, v in model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
            ]
            params_name = [
                k for k, v in model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k
            ]
        print('========trainable params============', params_name)

    # print trainable params's information
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print('The number of Total Trainable Parameters------------------:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices) # 模型并行化

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    for iteration in tqdm(range(total_iterations + 1)):
        if eval_iterations is not None and iteration % eval_iterations == 0:
            evaluate(model.module, args, val_preprocess)

        # training  finetune
        if iteration % num_batches == 0:

            if args.few_shot>0:  # default is -1
                data_iter = iter(few_shot_data_loader)
            else:
                data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data
        if args.train_dataset == 'ImageNet':
            try:
                train_batch = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                train_batch = next(data_iter)
            images, labels = train_batch["images"], train_batch["labels"]
        else:
            try:
                images, labels = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()

        # -- get text embedding --
        if args.train_mode != "text":
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)
        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())


    # save experts' frequency of activation
    with open(frozen_path, "a") as file:
        for i in range(12):
            visual_choose_map = model.module.visual.transformer.resblocks[i].choose_map_image
            text_choose_map = model.module.transformer.resblocks[i].choose_map_text
            top_values_v, top_indices_v = torch.topk(visual_choose_map, 2)
            top_values_t, top_indices_t = torch.topk(text_choose_map, 2)

            for j in range(len(top_indices_v)):
                item1 = 'visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i,top_indices_v[j])
                item2 = 'visual.transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i,top_indices_v[j])
                item3 = 'visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i,top_indices_v[j])
                item4 = 'visual.transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i,top_indices_v[j])
                file.write(item1 + "\n")
                file.write(item2 + "\n")
                file.write(item3 + "\n")
                file.write(item4 + "\n")
            for k in range(len(top_indices_t)):
                item1 = 'transformer.resblocks.{}.adaptmlp_list.{}.down_proj.weight'.format(i, top_indices_t[k])
                item2 = 'transformer.resblocks.{}.adaptmlp_list.{}.down_proj.bias'.format(i, top_indices_t[k])
                item3 = 'transformer.resblocks.{}.adaptmlp_list.{}.up_proj.weight'.format(i, top_indices_t[k])
                item4 = 'transformer.resblocks.{}.adaptmlp_list.{}.up_proj.bias'.format(i, top_indices_t[k])
                file.write(item1 + "\n")
                file.write(item2 + "\n")
                file.write(item3 + "\n")
                file.write(item4 + "\n")
        print('=======================bingo!=============================')


    # Saving model
    if args.save is not None:
        to_save_model = model.module
        # to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)
