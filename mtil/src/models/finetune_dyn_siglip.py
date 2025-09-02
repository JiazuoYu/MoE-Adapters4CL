# --------------------------------------------------------
# Training code for MoE-Adapters++, 
# compatible with MoE-Adapters
# --------------------------------------------------------
# import copy
import os
# import numpy as np
# import torch.distributed as dist
from safetensors import safe_open
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import siglip.siglip.modeling_siglip as siglip
from siglip.siglip.processing_siglip import SiglipProcessor
from siglip.custom_siglip import CustomSiglipModel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from transformers import logging as transformer_logging
# Ignore transformer warning
transformer_logging.set_verbosity_error()
import clip.clip as clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# import pandas as pd
from .. import datasets, templates, utils
# from .evaluation import evaluate, zeroshot_classifier
# from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation
# from clip.loss import total_loss
from siglip.loss import total_loss


# clip
import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, InterpolationMode
from tqdm import tqdm

# from .model import build_model
# from .tokenizer import SimpleTokenizer as _Tokenizer

train_iteration = None


def finetune_dyn_siglip(args):
    # init TensorBoard SummaryWriter
    writer = None
    if args.log_dir is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
    print("LR: ", args.lr, "RD_LR: ", args.lr_ae)

    # dyn_moe layer
    print('text use Dynamic MoE_Adapters in layers:', args.use_dyn_moe_layer_list_text)  # not use in text encoder
    print('visual use Dynamic MoE_Adapters in layers:', args.use_dyn_moe_layer_list_visual)

    # TODO: check the model
    if args.load is not None and args.repeat_train:  # continual learning && use dyn_moe
        text_expert_num, image_expert_num = get_experts_and_router_num_siglip(args.load, args)
        args.text_expert_num_list = text_expert_num
        args.image_expert_num_list = image_expert_num
        print('use Dynamic MoE_Adapters for continual learning')
        print('text_expert_num in all layers:', text_expert_num)
        print('image_expert_num in all layers:', image_expert_num)
        # useful_params = get_useful_params_continual(args)
        useful_params = get_only_one_useful_params_continual(args)
        
        print(f"build pre-trained model config: {args.model}")
        hf_config = AutoConfig.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
        model = CustomSiglipModel.from_pretrained(
            pretrained_model_name_or_path=args.load,
            args=args,
            config=hf_config,
            ignore_mismatched_sizes=True,
            )
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
        
        print(f"build pre-trained model config: {args.model}")
        hf_config = AutoConfig.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
        model = CustomSiglipModel.from_pretrained(pretrained_model_name_or_path="/home/dhw/HZC_workspace/Models/siglip-base-patch16-224",
                                                args=args,
                                                config=hf_config,
                                                ignore_mismatched_sizes=True,
                                                )
        
    attn_mask = model.build_attention_mask()

    # model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, args=args)  # model='ViT-B/16'
    processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")  # 暂时使用固定路径的base模型
    tokenizer = AutoTokenizer.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
    # siglip.SiglipModel.from_pretrained

    #  train_preprocess is_train=True val_preprocess is_train=False
    # if args.load is not None and args.repeat_train is True:  # continual learning
    #     print('[model state] continual training')
    #     utils.torch_load(model, args.load)

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
    # expert block name: text_model.encoder.layers.[block_num].adaptmlp_list.[experts_num].up_proj
    # router list name: text_model.encoder.layers.[block_num].w_noise_list.[router_num]

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
    # texts = clip.tokenize(texts).cuda()
    # texts = tokenizer(texts).cuda() 
    texts = processor(text=texts,padding="max_length")["input_ids"].cuda()  # siglip

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
        
        output = model(
            input_ids=texts,
            pixel_values=images,
            attention_mask=attn_mask,
            return_loss=False,
            return_dict=True,
            )
        
        """def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SiglipOutput]:
        
        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_essh-add -lmbeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
        """
        logits_per_image = output["logits_per_image"]
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
        
        # # -- get text embedding --
        # embeddings = model.get_text_features(None, texts)
        # embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # # -- get image embedding --
        # out, _ = model(images, None)
        # out = out / out.norm(dim=-1, keepdim=True)
        # # -- cross entropy loss --
        # logits_per_image = logit_scale.exp() * out @ embeddings.t()
        # loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)  # ce_loss
        
        # record ce_loss
        ce_loss_to_tensorboard(loss, writer, iteration)
        # get mse_loss for LEAS & DEeC
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

    # save activated experts & models
    dyn_save_model_siglip(args, model)


# def _transform(n_px: int, is_train: bool):
#     normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     if is_train:
#         return Compose([
#             RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
#             clip._convert_to_rgb,
#             ToTensor(),
#             normalize,
#         ])
#     else:
#         return Compose([
#             Resize(n_px, interpolation=InterpolationMode.BICUBIC),
#             CenterCrop(n_px),
#             clip._convert_to_rgb,
#             ToTensor(),
#             normalize,
#         ])
        
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
            text_adaptmlp_key = f"text_model.encoder.layers.{i}.activated_experts_num"
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
            image_adaptmlp_key = f"vision_model.encoder.layers.{i}.activated_experts_num"
            image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key]
            current_image_experts_num = state_dict[image_matching_keys[0]][0].item()
            image_expert_counts.append(int(current_image_experts_num))
            if current_image_experts_num == [1.0]:
                image_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        else:
            image_expert_counts.append(0)
    return text_expert_counts, image_expert_counts


def get_experts_and_router_num_siglip(model_path, args):
    """从 HuggingFace safetensors 格式检查点读取专家和路由器数量"""
    # 确定权重文件路径
    if os.path.isdir(model_path):
        # 查找目录中的 safetensors 文件
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        # 通常只有一个主要的 safetensors 文件
        safetensors_file = os.path.join(model_path, safetensors_files[0])
    else:
        safetensors_file = model_path

    # 使用 safetensors 加载模型权重
    with safe_open(safetensors_file, framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    text_expert_counts = []
    image_expert_counts = []

    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            # 构造需要检查的关键字
            text_adaptmlp_key = f"text_model.encoder.layers.{i}.activated_experts_num"
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
            image_adaptmlp_key = f"vision_model.encoder.layers.{i}.activated_experts_num"
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
            text_choose_map = model.module.text_model.encoder.layers[i].choose_map_text
            text_rate = F.normalize(text_choose_map, p=1.0, dim=0)  # get rate
            model.module.text_model.encoder.layers[i].update_freq_activated_experts(text_rate)  # save to model
            # update avg & std of RD_loss
            _update_mse_loss_info_text(model, args, i)
            # update activated expert_nums and routers
            text_activated_expert = torch.sum(model.module.text_model.encoder.layers[i].experts_mask)
        else:
            text_activated_expert = 0
        if args.use_LEAS_list_text[i]:
            # update avg & std of RD_loss
            _update_mse_loss_info_text(model, args, i)
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            # if args.single_router is False:
            # image activated record
            visual_choose_map = model.module.vision_model.encoder.layers[i].choose_map_image
            image_rate = F.normalize(visual_choose_map, p=1.0, dim=0)  # get rate
            model.module.vision_model.encoder.layers[i].update_freq_activated_experts(image_rate)  # save to model
            # update avg & std of RD_loss
            _update_mse_loss_info_visual(model, args, i)
            # update activated expert_nums and routers
            visual_activated_expert = torch.sum(model.module.vision_model.encoder.layers[i].experts_mask)
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
        
def dyn_save_model_siglip(args, model):
    # 记录专家使用频率（与原函数相同）
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i]:
            text_choose_map = model.module.text_model.encoder.layers[i].choose_map_text
            text_rate = F.normalize(text_choose_map, p=1.0, dim=0)
            model.module.text_model.encoder.layers[i].update_freq_activated_experts(text_rate)
            _update_mse_loss_info_text(model, args, i)
            text_activated_expert = torch.sum(model.module.text_model.encoder.layers[i].experts_mask)
        else:
            text_activated_expert = 0
        if args.use_LEAS_list_text[i]:
            _update_mse_loss_info_text(model, args, i)
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            visual_choose_map = model.module.vision_model.encoder.layers[i].choose_map_image
            image_rate = F.normalize(visual_choose_map, p=1.0, dim=0)
            model.module.vision_model.encoder.layers[i].update_freq_activated_experts(image_rate)
            _update_mse_loss_info_visual(model, args, i)
            visual_activated_expert = torch.sum(model.module.vision_model.encoder.layers[i].experts_mask)
        else:
            visual_activated_expert = 0
        if args.use_LEAS_list_visual[i]:
            _update_mse_loss_info_visual(model, args, i)
        
        print(f"Layer {i}, text expert_num: {text_activated_expert}, visual expert_num: {visual_activated_expert}")
    
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
        
        # print(f"Model saved in PyTorch format to {torch_path}")

def _update_mse_loss_info_visual(model, args, i):
    if args.use_LEAS_to_eval:
        cut_off_rate_new = args.cut_off_rate_visual
        cut_off_rate_frozen = args.cut_off_rate_visual
        if args.repeat_train is False:  # init
            cut_off_rate_new = args.cut_off_rate_visual
        model.module.vision_model.encoder.layers[i].update_mse_loss_avg_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )
        model.module.vision_model.encoder.layers[i].update_mse_loss_std_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )

def _update_mse_loss_info_text(model, args, i):
    if args.use_LEAS_to_eval:
        cut_off_rate_new = args.cut_off_rate_text
        cut_off_rate_frozen = args.cut_off_rate_text
        if args.repeat_train is False:  # init
            cut_off_rate_new = args.cut_off_rate_text
        model.module.text_model.encoder.layers[i].update_mse_loss_avg_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )
        model.module.text_model.encoder.layers[i].update_mse_loss_std_list(
            cut_off_rate_new=cut_off_rate_new,
            cut_off_rate_frozen=cut_off_rate_frozen,
        )


def dyn_get_mse_loss_text(model, text_experts_num, args):
    mse_loss = torch.tensor(0.0, device="cuda:0")
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            if model.module.text_model.encoder.layers[i].expansion_flag:
                if args.repeat_train:
                    mse_loss += model.module.text_model.encoder.layers[i].mse_loss_list[int(text_experts_num[i])]
                else:  # init
                    mse_loss += model.module.text_model.encoder.layers[i].mse_loss
    return mse_loss


def dyn_get_mse_loss_image(model, image_experts_num, args):
    mse_loss = torch.tensor(0.0, device="cuda:0")
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            if model.module.vision_model.encoder.layers[i].expansion_flag:
                if args.repeat_train:
                    mse_loss += model.module.vision_model.encoder.layers[i].mse_loss_list[int(image_experts_num[i])]
                else:  # init
                    mse_loss += model.module.vision_model.encoder.layers[i].mse_loss
    return mse_loss


def get_useful_params_init(args):
    useful_list = []
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            for j in range(args.init_expert_num):
                # text-expert
                useful_list.append(f"text_model.encoder.layers.{i}.auto_encoder_list.{j}.")
                useful_list.append(f"text_model.encoder.layers.{i}.adaptmlp_list.{j}.")
            if args.single_router:
                # text-router
                useful_list.append(f"text_model.encoder.layers.{i}.w_noise1")
                useful_list.append(f"text_model.encoder.layers.{i}.router1")
            else:
                # text-router
                useful_list.append(f"text_model.encoder.layers.{i}.w_noise_list.{args.task_id}")
                useful_list.append(f"text_model.encoder.layers.{i}.router_list.{args.task_id}")
    for i in range(args.vision_layer):  
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            for j in range(args.init_expert_num):
                # visual-expert
                useful_list.append(f"vision_model.encoder.layers.{i}.auto_encoder_list.{j}.")
                useful_list.append(f"vision_model.encoder.layers.{i}.adaptmlp_list.{j}.")
            if args.single_router:
                # visual-router
                useful_list.append(f"vision_model.encoder.layers.{i}.w_noise1")
                useful_list.append(f"vision_model.encoder.layers.{i}.router1")  
            else:
                # visual-router
                useful_list.append(f"vision_model.encoder.layers.{i}.w_noise_list.{args.task_id}")
                useful_list.append(f"vision_model.encoder.layers.{i}.router_list.{args.task_id}")
            
    return useful_list


def get_only_one_useful_params_continual(args):
    useful_list = []
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            text_idx = args.text_expert_num_list[i]
            # text-expert
            useful_list.append(f"text_model.encoder.layers.{i}.auto_encoder_list.{text_idx}.")
            useful_list.append(f"text_model.encoder.layers.{i}.adaptmlp_list.{text_idx}.")
        if args.single_router is False:    
            # text-router
            useful_list.append(f"text_model.encoder.layers.{i}.w_noise_list.{args.task_id}")
            useful_list.append(f"text_model.encoder.layers.{i}.router_list.{args.task_id}")
        else:
            # text-router
            useful_list.append(f"text_model.encoder.layers.{i}.w_noise1")
            useful_list.append(f"text_model.encoder.layers.{i}.router1")
            
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            image_idx = args.image_expert_num_list[i]
            # visual-expert
            useful_list.append(f"vision_model.encoder.layers.{i}.auto_encoder_list.{image_idx}.")
            useful_list.append(f"vision_model.encoder.layers.{i}.adaptmlp_list.{image_idx}.")
        if args.single_router is False:
            # visual-router
            useful_list.append(f"vision_model.encoder.layers.{i}.w_noise_list.{args.task_id}")
            useful_list.append(f"vision_model.encoder.layers.{i}.router_list.{args.task_id}")
        else:
            # visual-router
            useful_list.append(f"vision_model.encoder.layers.{i}.w_noise1")
            useful_list.append(f"vision_model.encoder.layers.{i}.router1")
            
    return useful_list


def loss_to_tensorboard(model, writer, idx, args, image_expert_num, text_expert_num):
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i] or args.use_LEAS_list_text[i]:
            text_mse_loss_list = model.module.text_model.encoder.layers[i].mse_loss_list[:text_expert_num[i] + 1]
            # 记录损失到 TensorBoard
            text_loss_dict = {f'Loss of RD{i}': loss for i, loss in enumerate(text_mse_loss_list)}
            writer.add_scalars(f'Text_losses_of_layer{i}', text_loss_dict, idx)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i] or args.use_LEAS_list_visual[i]:
            visual_mse_loss_list = model.module.vision_model.encoder.layers[i].mse_loss_list[:image_expert_num[i] + 1]
            # 记录损失到 TensorBoard
            visual_loss_dict = {f'Loss of RD{i}': loss for i, loss in enumerate(visual_mse_loss_list)}
            writer.add_scalars(f'Visual_losses_of_layer{i}', visual_loss_dict, idx)


def ce_loss_to_tensorboard(ce_loss, writer, idx):
    # 记录CE损失到 TensorBoard
    writer.add_scalar(f'CE_Loss', ce_loss, idx)


def _set_model_iter(model, iteration, args):
    for i in range(args.text_layer):
        model.module.text_model.encoder.layers[i].set_iteration(iteration)
    for i in range(args.vision_layer):
        model.module.vision_model.encoder.layers[i].set_iteration(iteration)