# --------------------------------------------------------
# Training code for MoE-Adapters++, 
# compatible with MoE-Adapters
# --------------------------------------------------------
import copy
import os
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import clip.clip as clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation
from clip.loss import total_loss

train_iteration = None


def finetune_dyn(args):
    # init TensorBoard SummaryWriter
    writer = None
    if args.log_dir is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
    print("LR: ", args.lr, "RD_LR: ", args.lr_ae)

    # dyn_moe layer
    print('text use Dynamic MoE_Adapters in layers:', args.use_dyn_moe_layer_list_text)  # not use in text encoder
    print('visual use Dynamic MoE_Adapters in layers:', args.use_dyn_moe_layer_list_visual)

    # check the model
    if args.load is not None and args.repeat_train:  # continual learning && use dyn_moe
        text_expert_num, image_expert_num = get_experts_and_router_num(args.load, args)
        args.text_expert_num_list = text_expert_num
        args.image_expert_num_list = image_expert_num
        print('use Dynamic MoE_Adapters for continual learning')
        print('text_expert_num in all layers:', text_expert_num)
        print('image_expert_num in all layers:', image_expert_num)
        # useful_params = get_useful_params_continual(args)
        useful_params = get_only_one_useful_params_continual(args)
        # print('router_num:', router_num)
    else:  # first train && use dyn_moe
        text_expert_num = [args.init_expert_num if dyn or id_det else 0
                            for dyn, id_det in zip(args.use_dyn_moe_layer_list_text, args.use_LEAS_list_text)]
        image_expert_num = [args.init_expert_num if dyn or id_det else 0
                            for dyn, id_det in zip(args.use_dyn_moe_layer_list_visual, args.use_LEAS_list_visual)]

        # image_expert_num = [args.init_expert_num if value else 0 for value in args.use_dyn_moe_layer_list_visual]
        args.text_expert_num_list = text_expert_num
        args.image_expert_num_list = image_expert_num
        print(f'use Dynamic MoE_Adapters for continual learning, init the model with {int(args.init_expert_num)} experts')
        print('[model state] init training')
        useful_params = get_useful_params_init(args)

    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'

    #  train_preprocess is_train=True val_preprocess is_train=False
    if args.load is not None and args.repeat_train is True:  # continual learning
        print('[model state] continual training')
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
        print(f'[data model] few-shot-mode, {args.few_shot}-shot')
        # few-shot
        few_shot_data = {}  # create few_shot data

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
        print("batches:", len(few_shot_data_loader))

    if args.few_shot > 0:
        num_batches = len(few_shot_data_loader)
    else:  # full_shot
        num_batches = len(dataset.train_loader)
    if args.epochs is not None:  # # False
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations  # 1000
    if args.eval_every_epoch:  # False
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval  # none
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params, Dynamic MoE_Adapters
    print("[Training mode] Dynamic Moe-Adapters: only adapters(experts) & routers")
    # expert block name: transformer.resblocks.[block_num].adaptmlp_list.[experts_num].up_proj
    # router list name: transformer.resblocks.[block_num].w_noise_list.[router_num]

    for k, v in model.named_parameters():  # frozen params
        # if not any(exclude_str in k for exclude_str in exclude_list):
        #     v.requires_grad = False
        if any(k.startswith(s) for s in useful_params):
            v.requires_grad = True
        else:
            v.requires_grad = False

        # print('frozen mode========trainable params============', params_name)
        # print('frozen mode========frozen params of trainable params============', frozen_list)
        # params = [
        #     v for k, v in model.named_parameters() if any(include_str in k for include_str in exclude_list)
        # ]
    params = [
        v for k, v in model.named_parameters() if any(k.startswith(s) for s in useful_params) and ".auto_encoder_list." not in k
    ]
    params_rd = [
        v for k, v in model.named_parameters() if any(k.startswith(s) for s in useful_params) and ".auto_encoder_list." in k
    ]
    params_name = [
        k for k, v in model.named_parameters() if any(k.startswith(s) for s in useful_params)
    ]
    # print('===========trainable params============\n', params_name)
    # print('===========trainable params============', params_name)

    # print trainable params' information
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print('The number of Total Trainable Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

    # optimizer
    # optimizer = torch.optim.AdamW(
    #     params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    # )
    optimizer = torch.optim.AdamW([
        {'params': params, 'lr': args.lr, 'weight_decay': args.wd, 'betas': (0.9, args.beta2)},
        {'params': params_rd, 'lr': args.lr_ae, 'weight_decay': args.wd, 'betas': (0.9, args.beta2)}
    ])
    scheduler = utils.cosine_lr(
        optimizer, [args.lr, args.lr_ae], args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)

    # if args.dyn_moe is False:
    #     model = torch.nn.DataParallel(model, device_ids=devices)  # 模型并行化
    # else:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model = torch.nn.DataParallel(model, device_ids=devices)  # 模型并行化

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    for iteration in tqdm(range(total_iterations + 1), ncols=50):
        # if eval_iterations is not None and iteration % eval_iterations == 0:
        #     evaluate(model.module, args, val_preprocess)
        _set_model_iter(model, iteration, args)  # set model iteration

        # training  finetune
        if iteration % num_batches == 0:

            if args.few_shot > 0:  # default is -1
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
        embeddings = model(None, texts)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out, _ = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)
        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)  # ce_loss
        # record ce_loss
        ce_loss_to_tensorboard(loss, writer, iteration)
        # get mse_loss
        mse_loss_image = dyn_get_mse_loss_image(model, image_expert_num, args)
        mse_loss_text = dyn_get_mse_loss_text(model, text_expert_num, args)
        mse_loss = mse_loss_text + mse_loss_image
        # get total_loss
        loss = total_loss(loss, mse_loss, args)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
        if writer is not None:
            loss_to_tensorboard(model, writer, iteration, args, image_expert_num, text_expert_num)
            discrepancy_to_tensorboard(model, writer, iteration, args)

    # save activated experts & models
    dyn_save_model(args, model)


# def _transform(n_px: int, is_train: bool):
#     normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     if is_train:
#         return Compose([
#             RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
#             _convert_to_rgb,
#             ToTensor(),
#             normalize,
#         ])
#     else:
#         return Compose([
#             Resize(n_px, interpolation=InterpolationMode.BICUBIC),
#             CenterCrop(n_px),
#             _convert_to_rgb,
#             ToTensor(),
#             normalize,
#         ])

def get_experts_and_router_num(model_path, args):
    # 加载保存的模型权重
    checkpoint = torch.load(model_path)

    # 提取state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    text_expert_counts = []
    image_expert_counts = []

    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            # 构造需要检查的关键字
            text_adaptmlp_key = f"transformer.resblocks.{i}.activated_experts_num"
            # 提取匹配的参数名称
            text_matching_keys = [key for key in state_dict.keys() if
                                    text_adaptmlp_key in key and "visual" not in key]
            current_text_experts_num = state_dict[text_matching_keys[0]][0].item()
            text_expert_counts.append(int(current_text_experts_num))
            if current_text_experts_num == [1.0]:
                text_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        else:
            text_expert_counts.append(0)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            image_adaptmlp_key = f"visual.transformer.resblocks.{i}.activated_experts_num"
            image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key]
            current_image_experts_num = state_dict[image_matching_keys[0]][0].item()
            image_expert_counts.append(int(current_image_experts_num))
            if current_image_experts_num == [1.0]:
                image_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        else:
            image_expert_counts.append(0)
    return text_expert_counts, image_expert_counts


def dyn_save_model(args, model):
    # get freq of experts
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i]:
            # if args.single_router is False:
            # text activated record
            text_choose_map = model.module.transformer.resblocks[i].choose_map_text
            text_rate = F.normalize(text_choose_map, p=1.0, dim=0)  # get rate
            model.module.transformer.resblocks[i].update_freq_activated_experts(text_rate)  # save to model
            # update avg & std of RD_loss
            _update_mse_loss_info_text(model, args, i)
            # update activated expert_nums and routers
            text_activated_expert = torch.sum(model.module.transformer.resblocks[i].experts_mask)
        else:
            text_activated_expert = 0
        if args.use_LEAS_list_text[i]:
            # update avg & std of RD_loss
            _update_mse_loss_info_text(model, args, i)
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            # if args.single_router is False:
            # image activated record
            visual_choose_map = model.module.visual.transformer.resblocks[i].choose_map_image
            image_rate = F.normalize(visual_choose_map, p=1.0, dim=0)  # get rate
            model.module.visual.transformer.resblocks[i].update_freq_activated_experts(image_rate)  # save to model
            # update avg & std of RD_loss
            _update_mse_loss_info_visual(model, args, i)
            # update activated expert_nums and routers
            visual_activated_expert = torch.sum(model.module.visual.transformer.resblocks[i].experts_mask)
        else:
            visual_activated_expert = 0
        if args.use_LEAS_list_visual[i]:
            # update avg & std of RD_loss
            _update_mse_loss_info_visual(model, args, i)
        
        print(f"Layer {i}, text expert_num: {text_activated_expert}, visual expert_num: {visual_activated_expert}")
    # Saving model
    if args.save is not None:
        to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)

def _update_mse_loss_info_visual(model, args, i):
    if args.use_LEAS_to_eval:
        cut_off_rate_new = args.cut_off_rate_visual
        cut_off_rate_frozen = args.cut_off_rate_visual
        if args.repeat_train is False:  # init
            cut_off_rate_new = args.cut_off_rate_visual
        model.module.visual.transformer.resblocks[i].update_mse_loss_avg_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )
        model.module.visual.transformer.resblocks[i].update_mse_loss_std_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )

def _update_mse_loss_info_text(model, args, i):
    if args.use_LEAS_to_eval:
        cut_off_rate_new = args.cut_off_rate_text
        cut_off_rate_frozen = args.cut_off_rate_text
        if args.repeat_train is False:  # init
            cut_off_rate_new = args.cut_off_rate_text
        model.module.transformer.resblocks[i].update_mse_loss_avg_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )
        model.module.transformer.resblocks[i].update_mse_loss_std_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )


def dyn_get_mse_loss_text(model, text_experts_num, args):
    mse_loss = torch.tensor(0.0, device="cuda:0")
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            if model.module.transformer.resblocks[i].expansion_flag:
                if args.repeat_train:
                    mse_loss += model.module.transformer.resblocks[i].mse_loss_list[int(text_experts_num[i])]
                else:  # init
                    mse_loss += model.module.transformer.resblocks[i].mse_loss
    return mse_loss


def dyn_get_mse_loss_image(model, image_experts_num, args):
    mse_loss = torch.tensor(0.0, device="cuda:0")
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            if model.module.visual.transformer.resblocks[i].expansion_flag:
                if args.repeat_train:
                    mse_loss += model.module.visual.transformer.resblocks[i].mse_loss_list[int(image_experts_num[i])]
                else:  # init
                    mse_loss += model.module.visual.transformer.resblocks[i].mse_loss
    return mse_loss


def get_useful_params_init(args):
    useful_list = []
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            for j in range(args.init_expert_num):
                # text-expert
                useful_list.append(f"transformer.resblocks.{i}.auto_encoder_list.{j}.")
                useful_list.append(f"transformer.resblocks.{i}.adaptmlp_list.{j}.")
            if args.single_router:
                # text-router
                useful_list.append(f"transformer.resblocks.{i}.w_noise1")
                useful_list.append(f"transformer.resblocks.{i}.router1")
            else:
                # text-router
                useful_list.append(f"transformer.resblocks.{i}.w_noise_list.{args.task_id}")
                useful_list.append(f"transformer.resblocks.{i}.router_list.{args.task_id}")
    for i in range(args.vision_layer):  
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            for j in range(args.init_expert_num):
                # visual-expert
                useful_list.append(f"visual.transformer.resblocks.{i}.auto_encoder_list.{j}.")
                useful_list.append(f"visual.transformer.resblocks.{i}.adaptmlp_list.{j}.")
            if args.single_router:
                # visual-router
                useful_list.append(f"visual.transformer.resblocks.{i}.w_noise1")
                useful_list.append(f"visual.transformer.resblocks.{i}.router1")  
            else:
                # visual-router
                useful_list.append(f"visual.transformer.resblocks.{i}.w_noise_list.{args.task_id}")
                useful_list.append(f"visual.transformer.resblocks.{i}.router_list.{args.task_id}")
                
    return useful_list


def get_only_one_useful_params_continual(args):
    useful_list = []
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            text_idx = args.text_expert_num_list[i]
            # text-expert
            useful_list.append(f"transformer.resblocks.{i}.auto_encoder_list.{text_idx}.")
            useful_list.append(f"transformer.resblocks.{i}.adaptmlp_list.{text_idx}.")
        if args.single_router is False:    
            # text-router
            useful_list.append(f"transformer.resblocks.{i}.w_noise_list.{args.task_id}")
            useful_list.append(f"transformer.resblocks.{i}.router_list.{args.task_id}")
        else:
            # text-router
            useful_list.append(f"transformer.resblocks.{i}.w_noise1")
            useful_list.append(f"transformer.resblocks.{i}.router1")
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            image_idx = args.image_expert_num_list[i]
            # visual-expert
            useful_list.append(f"visual.transformer.resblocks.{i}.auto_encoder_list.{image_idx}.")
            useful_list.append(f"visual.transformer.resblocks.{i}.adaptmlp_list.{image_idx}.")
        if args.single_router is False:
            # visual-router
            useful_list.append(f"visual.transformer.resblocks.{i}.w_noise_list.{args.task_id}")
            useful_list.append(f"visual.transformer.resblocks.{i}.router_list.{args.task_id}")
        else:
            # visual-router
            useful_list.append(f"visual.transformer.resblocks.{i}.w_noise1")
            useful_list.append(f"visual.transformer.resblocks.{i}.router1")
            
    return useful_list


def loss_to_tensorboard(model, writer, idx, args, image_expert_num, text_expert_num):
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            text_mse_loss_list = model.module.transformer.resblocks[i].mse_loss_list[:text_expert_num[i] + 1]
            # 记录损失到 TensorBoard
            text_loss_dict = {f'Loss of RD{i}': loss for i, loss in enumerate(text_mse_loss_list)}
            writer.add_scalars(f'Text_losses_of_layer{i}', text_loss_dict, idx)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            visual_mse_loss_list = model.module.visual.transformer.resblocks[i].mse_loss_list[:image_expert_num[i] + 1]
            # 记录损失到 TensorBoard
            visual_loss_dict = {f'Loss of RD{i}': loss for i, loss in enumerate(visual_mse_loss_list)}
            writer.add_scalars(f'Visual_losses_of_layer{i}', visual_loss_dict, idx)
            
def discrepancy_to_tensorboard(model, writer, idx, args):
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            text_eval_discrepancy_list = model.module.transformer.resblocks[i].eval_discrepancy_list
            if len(text_eval_discrepancy_list) > 0:
                writer.add_scalar(f'Text_Train_Discrepancy_of_layer{i}', max(text_eval_discrepancy_list), idx)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            if model.module.visual.transformer.resblocks[i].reconginition_layer:
                # 记录视觉层的训练不一致性
                visual_eval_discrepancy_list = model.module.visual.transformer.resblocks[i].current_discrepancy_train
                writer.add_scalar(f'Visual_Train_Discrepancy_of_layer{i}', visual_eval_discrepancy_list, idx)


def ce_loss_to_tensorboard(ce_loss, writer, idx):
    # 记录CE损失到 TensorBoard
    writer.add_scalar(f'CE_Loss', ce_loss, idx)
    
    
def _set_model_iter(model, iteration, args):
    for i in range(args.text_layer):
        model.module.transformer.resblocks[i].set_iteration(iteration)
    for i in range(args.vision_layer):
        model.module.visual.transformer.resblocks[i].set_iteration(iteration)
