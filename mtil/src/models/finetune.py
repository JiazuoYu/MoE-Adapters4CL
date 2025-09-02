import copy
import os
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode
import clip.clip as clip
import siglip.siglip.modeling_siglip as siglip
from siglip.siglip.processing_siglip import SiglipProcessor
from siglip.custom_siglip import CustomSiglipModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation
from clip.loss import total_loss
# from siglip.loss import total_loss_siglip
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from transformers import logging as transformer_logging
# Ignore transformer warning
transformer_logging.set_verbosity_error()


train_iteration = None


def finetune(args):
    # print('---1---',args.frozen_path)
    frozen_path = args.frozen_path
    if args.output_frozen_path is None:
        output_frozen_path = frozen_path
        print('output_frozen_path:', output_frozen_path)
    else:
        output_frozen_path = args.output_frozen_path
        print('output_frozen_path:', output_frozen_path)
    # 获取上级目录
    parent_dir = os.path.dirname(frozen_path)
    # 创建上级目录（如果不存在）
    os.makedirs(parent_dir, exist_ok=True)
    # 获取上级目录
    output_frozen_path_parent_dir = os.path.dirname(output_frozen_path)
    # 创建上级目录（如果不存在）
    os.makedirs(output_frozen_path_parent_dir, exist_ok=True)
    
    writer = None
    # 初始化 TensorBoard SummaryWriter
    if args.log_dir is not None:
        writer = SummaryWriter(log_dir=args.log_dir)

    # check the model
    if args.dyn_moe:
        if args.load is not None and args.repeat_train:  # continual learning && use dyn_moe
            text_expert_num, image_expert_num = get_experts_and_router_num(args.load, args)
            args.text_expert_num_list = text_expert_num
            args.image_expert_num_list = image_expert_num
            print('use dyn_moe for continual learning')
            print('text_expert_num in all layers:', text_expert_num)
            print('image_expert_num in all layers:', image_expert_num)
            # useful_params = get_useful_params_continual(args)
            useful_params = get_only_one_useful_params_continual(args)
            # print('router_num:', router_num)
        else:  # first train && use dyn_moe
            text_expert_num, image_expert_num = [args.init_expert_num] * args.vision_layer, [args.init_expert_num] * args.vision_layer
            args.text_expert_num_list = [args.init_expert_num] * args.vision_layer
            args.image_expert_num_list = [args.init_expert_num] * args.vision_layer
            print(f'use dyn_moe for continual learning, init the model with {int(args.init_expert_num)} experts')
            useful_params = get_useful_params_init(args)
            
    if args.model == "Siglip-B-14-224":
        # build model
        if args.load is not None and args.repeat_train:  # continual learning
            print(f"build pre-trained model config: {args.model}")
            hf_config = AutoConfig.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
            model = CustomSiglipModel.from_pretrained(
            pretrained_model_name_or_path=args.load,
            args=args,
            config=hf_config,
            ignore_mismatched_sizes=True,
            )
        else:
            print(f"build pre-trained model config: {args.model}")
            hf_config = AutoConfig.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
            model = CustomSiglipModel.from_pretrained(
                pretrained_model_name_or_path="/home/dhw/HZC_workspace/Models/siglip-base-patch16-224",
                args=args,
                config=hf_config,
                ignore_mismatched_sizes=True,
                )
        # build tokenizer,processor
        attn_mask = model.build_attention_mask()
        # model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'
        processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")  # 暂时使用固定路径的base模型
        tokenizer = AutoTokenizer.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")

        # prepare dataset
        dataset_class = getattr(datasets, args.train_dataset)
        image_zise = [processor.image_processor.size["height"], processor.image_processor.size["width"]]  
        dataset = dataset_class(
        # train_preprocess,
        # processor.image_processor,
        build_siglip_transform(image_size=image_zise, is_train=True),
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )
        
    else: 
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'
        print('[model state] build clip model')
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
    # if args.dyn_moe:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)

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
    # for name, _ in model.named_parameters():
    #     print(name)

    # get params, original
    if args.train_mode == "adapter" and args.dyn_moe is not True:  # only train adapter
        print("[Training mode] Moe-Adapters: only adapters(experts) & routers")
        # expert block name: transformer.resblocks.[block_num].adaptmlp_list.[experts_num].up_proj
        # router list name: transformer.resblocks.[block_num].w_noise_list.[router_num]
        for k, v in model.named_parameters():  # frozen params
            if "adaptmlp" not in k and "router" not in k and "noise" not in k:
                v.requires_grad = False

        if args.frozen:  # frozen-strategy
            print('-----------------frozen-----------------')
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
        # print('===========trainable params============\n')
        # for i in params_name:
        #     print(i)
        # print('===========trainable params============', params_name)

    else:  # use dyn_moe, 废弃
        print("[Training mode] Dyn-Moe-Adapters: only adapters(experts) & routers")
        # expert block name: transformer.resblocks.[block_num].adaptmlp_list.[experts_num].up_proj
        # router list name: transformer.resblocks.[block_num].w_noise_list.[router_num]
        exclude_list = ["adaptmlp",
                        "router_list",
                        "gates_threshold",
                        "sim_matrix",
                        "w_noise_list",
                        # "representation_descriptor_list",
                        "encoder",
                        "decoder",
                        ]

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
            v for k, v in model.named_parameters() if any(k.startswith(s) for s in useful_params)
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

    # if args.dyn_moe is False:
    #     model = torch.nn.DataParallel(model, device_ids=devices)  # 模型并行化
    # else:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model = torch.nn.DataParallel(model, device_ids=devices)  # 模型并行化

    # text
    texts = [template(x) for x in dataset.classnames]
    if args.model == "Siglip-B-14-224":
        texts = processor(text=texts,padding="max_length")["input_ids"].cuda()  # siglip
    else:
        texts = clip.tokenize(texts).cuda()

    for iteration in tqdm(range(total_iterations + 1),ncols=50):
        # if eval_iterations is not None and iteration % eval_iterations == 0:
        #     evaluate(model.module, args, val_preprocess)
        if args.model == "Siglip-B-14-224":
            _set_model_iter_siglip(model, iteration, args)  # set model iteration

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
        print(texts.shape)
        print(images.shape)
        if args.model == "Siglip-B-14-224":
            output = model(
            input_ids=texts,
            pixel_values=images,
            attention_mask=attn_mask,
            return_loss=False,
            return_dict=True,
            )
            print(labels)
            logits_per_image = output["logits_per_image"]
            loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
            # # -- get text embedding --
            # embeddings = model.get_text_features(input_ids=texts, attention_mask=attn_mask, mode="text")[0]
            # embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            # # -- get image embedding --
            # image_features = model.get_image_features(images, mode="image")
            # image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            # # -- cross entropy loss --
            # logits_per_image = logit_scale.exp() * image_features @ embeddings.t()
            # loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
        else:
            rd_loss_text, rd_loss_image = 0., 0.
            # -- get text embedding --
            if args.train_mode != "text":
                embeddings = model(None, texts)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # if args.train_mode != "text":
            #     embeddings = model(None, texts)
            #     embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            #     if args.dyn_moe:
            #         rd_loss_text = dyn_get_rd_loss_text(model)

            # -- get image embedding --
            out, _ = model(images, None)
            out = out / out.norm(dim=-1, keepdim=True)
            # -- cross entropy loss --
            logits_per_image = logit_scale.exp() * out @ embeddings.t()
            loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)  # ce_loss

        if args.dyn_moe:
            ce_loss_to_tensorboard(loss, writer, iteration)
            rd_loss_image = dyn_get_rd_loss_image(model, image_expert_num, args)
            rd_loss_text = dyn_get_rd_loss_text(model, text_expert_num, args)
            rd_loss = rd_loss_text + rd_loss_image
            loss = total_loss(loss, rd_loss, args)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())

        # if args.dyn_moe:
        if writer is not None:
            loss_to_tensorboard(model, writer, iteration, image_expert_num, text_expert_num,args)

    # save activated experts & models
    if args.dyn_moe:
        dyn_save_model(args, model)
    else:
        if args.model == "Siglip-B-14-224":
            original_save_model_siglip(args, model, output_frozen_path)
        else:
            original_save_model(args, model, output_frozen_path)


# def dyn_get_loss(images, texts, model, args):
def original_get_experts_and_router_num(model_path, args):
    # 加载保存的模型权重
    checkpoint = torch.load(model_path)

    # 提取state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    text_expert_counts = []
    image_expert_counts = []

    for i in range(args.vision_layer):
        # 构造需要检查的关键字
        text_adaptmlp_key = f"transformer.resblocks.{i}.adaptmlp_list"
        image_adaptmlp_key = f"visual.transformer.resblocks.{i}.adaptmlp_list"
        up_proj_key = "up_proj"

        # 提取匹配的参数名称
        text_matching_keys = [key for key in state_dict.keys() if text_adaptmlp_key in key and up_proj_key in key and "visual" not in key]
        image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key and up_proj_key in key]

        if text_matching_keys:
            # 找到 adaptmlp_list 后面的数字并找到最大值
            max_number = max(int(key.split('adaptmlp_list.')[1].split('.')[0]) for key in text_matching_keys)
            text_expert_counts.append(max_number + 1)
        else:
            text_expert_counts.append(0)  # 如果没有匹配的参数，填入0
        if image_matching_keys:
            # 找到 adaptmlp_list 后面的数字并找到最大值
            max_number = max(int(key.split('adaptmlp_list.')[1].split('.')[0]) for key in image_matching_keys)
            image_expert_counts.append(max_number + 1)
        else:
            image_expert_counts.append(0)  # 如果没有匹配的参数，填入0

    return text_expert_counts, image_expert_counts


def get_experts_and_router_num(model_path, args):
    # 加载保存的模型权重
    checkpoint = torch.load(model_path)

    # 提取state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    text_expert_counts = []
    image_expert_counts = []

    for i in range(args.text_layer):
        # 构造需要检查的关键字
        text_adaptmlp_key = f"transformer.resblocks.{i}.activated_experts_num"
        # 提取匹配的参数名称
        text_matching_keys = [key for key in state_dict.keys() if
                              text_adaptmlp_key in key and "visual" not in key]
        current_text_experts_num = state_dict[text_matching_keys[0]][0].item()
        text_expert_counts.append(int(current_text_experts_num))
        
        if current_text_experts_num == [1.0]:
            text_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        
    for i in range(args.vision_layer):
        image_adaptmlp_key = f"visual.transformer.resblocks.{i}.activated_experts_num"
        image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key]
        current_image_experts_num = state_dict[image_matching_keys[0]][0].item()
        image_expert_counts.append(int(current_image_experts_num))
        if current_image_experts_num == [1.0]:
            image_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2

    return text_expert_counts, image_expert_counts


def dyn_save_model(args, model):
    # get freq of experts
    for i in range(args.text_layer):        
        # text activated record
        text_choose_map = model.module.transformer.resblocks[i].choose_map_text
        text_rate = F.normalize(text_choose_map, p=1.0, dim=0)  # get rate
        model.module.transformer.resblocks[i].update_freq_activated_experts(text_rate)  # save to model
        # update activated expert_nums and routers
        text_activated_expert = torch.sum(model.module.transformer.resblocks[i].experts_mask)
        print(f"Layer {i}, text expert_num: {text_activated_expert}")
    for i in range(args.vision_layer):
        # image activated record
        visual_choose_map = model.module.visual.transformer.resblocks[i].choose_map_image
        image_rate = F.normalize(visual_choose_map, p=1.0, dim=0)  # get rate
        model.module.visual.transformer.resblocks[i].update_freq_activated_experts(image_rate)  # save to model
        # update activated expert_nums and routers
        visual_activated_expert = torch.sum(model.module.visual.transformer.resblocks[i].experts_mask)
        print(f"Layer {i}, visual expert_num: {visual_activated_expert}")
    # Saving model
    if args.save is not None:
        to_save_model = model.module
        # 将数组添加到模型的 state_dict 中
        # model.state_dict()['extra_array'] = extra_array
        # to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)


def original_save_model(args, model, frozen_path):
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
        

def original_save_model_siglip(args, model, frozen_path):
    # save experts' frequency of activation
    with open(frozen_path, "a") as file:
        for i in range(args.text_layer):            
            text_choose_map = model.module.text_model.encoder.layers[i].choose_map_text
            top_values_t, top_indices_t = torch.topk(text_choose_map, 2)
            for k in range(len(top_indices_t)):
                item1 = 'text_model.encoder.layers.{}.adaptmlp_list.{}.down_proj.weight'.format(i, top_indices_t[k])
                item2 = 'text_model.encoder.layers.{}.adaptmlp_list.{}.down_proj.bias'.format(i, top_indices_t[k])
                item3 = 'text_model.encoder.layers.{}.adaptmlp_list.{}.up_proj.weight'.format(i, top_indices_t[k])
                item4 = 'text_model.encoder.layers.{}.adaptmlp_list.{}.up_proj.bias'.format(i, top_indices_t[k])
                file.write(item1 + "\n")
                file.write(item2 + "\n")
                file.write(item3 + "\n")
                file.write(item4 + "\n")
                      
        for i in range(args.vision_layer):
            visual_choose_map = model.module.vision_model.encoder.layers[i].choose_map_image
            top_values_v, top_indices_v = torch.topk(visual_choose_map, 2)
            for j in range(len(top_indices_v)):
                item1 = 'vision_model.encoder.layers.{}.adaptmlp_list.{}.down_proj.weight'.format(i, top_indices_v[j])
                item2 = 'vision_model.encoder.layers.{}.adaptmlp_list.{}.down_proj.bias'.format(i, top_indices_v[j])
                item3 = 'vision_model.encoder.layers.{}.adaptmlp_list.{}.up_proj.weight'.format(i, top_indices_v[j])
                item4 = 'vision_model.encoder.layers.{}.adaptmlp_list.{}.up_proj.bias'.format(i, top_indices_v[j])
                file.write(item1 + "\n")
                file.write(item2 + "\n")
                file.write(item3 + "\n")
                file.write(item4 + "\n")
            
        print('=======================bingo!=============================')

    # Saving model
    # 修改后的保存逻辑：支持HuggingFace模型保存
    if args.save is not None:
        to_save_model = model.module if hasattr(model, 'module') else model
        
        # 创建保存目录
        hf_path = os.makedirs(os.path.join(args.save, f"{args.train_dataset}"), exist_ok=True)
        
        # # 方案1：保存为PyTorch格式（兼容原始代码）
        # torch_path = os.path.join(args.save, f"{args.train_dataset}.pth")
        # torch.save(to_save_model.state_dict(), torch_path)
        
        # 方案2：保存为HuggingFace格式（新增）
        if hasattr(to_save_model, 'save_pretrained'):
            # hf_path = os.path.join(args.save, "hf_model")
            hf_path = os.path.join(args.save, f"{args.train_dataset}")
            to_save_model.save_pretrained(
                hf_path,
                safe_serialization=True  # 使用安全序列化格式
            )
            print(f"Model saved in HuggingFace format to {hf_path}")
            
            # 如果需要保存tokenizer（假设模型有相关属性）
            if hasattr(to_save_model, 'tokenizer'):
                tokenizer = to_save_model.tokenizer
                tokenizer.save_pretrained(hf_path)


def build_siglip_transform(image_size=224, is_train=False):
    """
    严格遵循SigLIP原始实现的图像预处理
    
    参数:
        image_size (int or tuple): 输出图像尺寸 (默认224)
        is_train (bool): 是否为训练模式
        
    返回:
        torchvision.transforms.Compose 对象
    """
    # 参数验证
    if isinstance(image_size, int):
        size = (image_size, image_size)
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        size = tuple(image_size)
    else:
        raise ValueError(
            f"image_size should be int or (height, width) tuple. Got {type(image_size)}"
        )

    # SigLIP官方归一化参数
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # 适用于3通道图像
        std=[0.5, 0.5, 0.5]
    )
    
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.Lambda(lambda x: x.convert('RGB')),  # 确保转换为RGB
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.Lambda(lambda x: x.convert('RGB')),  # 确保转换为RGB
            transforms.ToTensor(),
            normalize,
        ])
        

def dyn_get_rd_loss_text(model, text_experts_num, args):
    rd_loss = torch.tensor(0., device="cuda:0")  # TODO: 修复device编号问题，改成自动确定同一gpu
    for i in range(args.text_layer):
        if model.module.transformer.resblocks[i].expansion_flag and args.repeat_train:  # continual
            # rd_loss += model.module.transformer.resblocks[i].rd_loss
            rd_loss += model.module.transformer.resblocks[i].rd_loss_list[int(text_experts_num[i])]
        else:  # init
            rd_loss += model.module.transformer.resblocks[i].rd_loss
    return rd_loss


def dyn_get_rd_loss_image(model, image_experts_num, args):
    rd_loss = torch.tensor(0., device="cuda:0")  # TODO: 修复device编号问题，改成自动确定同一gpu
    for i in range(args.vision_layer):
        if model.module.visual.transformer.resblocks[i].expansion_flag and args.repeat_train: # continual
            rd_loss += model.module.visual.transformer.resblocks[i].rd_loss_list[int(image_experts_num[i])]
        else:  # init
            rd_loss += model.module.visual.transformer.resblocks[i].rd_loss
    return rd_loss


def get_useful_params_init(args):
    useful_list = []
    for i in range(args.text_layer):
        for j in range(args.init_expert_num):
            # text-expert
            useful_list.append(f"transformer.resblocks.{i}.representation_descriptor_list.{j}.")
            useful_list.append(f"transformer.resblocks.{i}.adaptmlp_list.{j}.")
        # text-router
        useful_list.append(f"transformer.resblocks.{i}.w_noise_list.{args.task_id}")
        useful_list.append(f"transformer.resblocks.{i}.router_list.{args.task_id}")
    for i in range(args.vision_layer):
        for j in range(args.init_expert_num):
            # visual-expert
            useful_list.append(f"visual.transformer.resblocks.{i}.representation_descriptor_list.{j}.")
            useful_list.append(f"visual.transformer.resblocks.{i}.adaptmlp_list.{j}.")
        # visual-router
        useful_list.append(f"visual.transformer.resblocks.{i}.w_noise_list.{args.task_id}")
        useful_list.append(f"visual.transformer.resblocks.{i}.router_list.{args.task_id}")
        

    return useful_list


def get_only_one_useful_params_continual(args):
    useful_list = []
    for i in range(args.text_layer):
        text_idx = args.text_expert_num_list[i]

        # text-expert
        useful_list.append(f"transformer.resblocks.{i}.representation_descriptor_list.{text_idx}.")
        useful_list.append(f"transformer.resblocks.{i}.adaptmlp_list.{text_idx}.")
        # text-router
        useful_list.append(f"transformer.resblocks.{i}.w_noise_list.{args.task_id}")
        useful_list.append(f"transformer.resblocks.{i}.router_list.{args.task_id}")

    for i in range(args.vision_layer):
        image_idx = args.image_expert_num_list[i]
        # visual-expert
        useful_list.append(f"visual.transformer.resblocks.{i}.representation_descriptor_list.{image_idx}.")
        useful_list.append(f"visual.transformer.resblocks.{i}.adaptmlp_list.{image_idx}.")
        # visual-router
        useful_list.append(f"visual.transformer.resblocks.{i}.w_noise_list.{args.task_id}")
        useful_list.append(f"visual.transformer.resblocks.{i}.router_list.{args.task_id}")
        
    return useful_list


def loss_to_tensorboard(model, writer, idx, image_expert_num, text_expert_num, args):
    for i in range(args.text_layer):
        text_rd_loss_list = model.module.transformer.resblocks[i].rd_loss_list[:text_expert_num[i] + 1]
        # 记录损失到 TensorBoard
        text_loss_dict = {f'Loss of RD{i}': loss for i, loss in enumerate(text_rd_loss_list)}
        writer.add_scalars(f'Text_losses_of_layer{i}', text_loss_dict, idx)
    for i in range(args.vision_layer):
        visual_rd_loss_list = model.module.visual.transformer.resblocks[i].rd_loss_list[:image_expert_num[i] + 1]
        # 记录损失到 TensorBoard
        visual_loss_dict = {f'Loss of RD{i}': loss for i, loss in enumerate(visual_rd_loss_list)}
        writer.add_scalars(f'Visual_losses_of_layer{i}', visual_loss_dict, idx)


def ce_loss_to_tensorboard(ce_loss, writer, idx):
    # 记录CE损失到 TensorBoard
    writer.add_scalar(f'CE_Loss', ce_loss, idx)


def _set_model_iter_siglip(model, iteration, args):
    for i in range(args.text_layer):
        model.module.text_model.encoder.layers[i].set_iteration(iteration)
    for i in range(args.vision_layer):
        model.module.vision_model.encoder.layers[i].set_iteration(iteration)