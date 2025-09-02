
import os

from safetensors import safe_open
import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, finetune, Autoencoder, Alexnet_FE, few_shot_AutoEncoder, AutoEncoder, dyn_evaluate, dyn_evaluate_siglip, finetune_dyn, finetune_dyn_siglip, build_siglip_transform
from .models.modeling import create_image_classifier

import siglip.siglip.modeling_siglip as siglip
from siglip.siglip.processing_siglip import SiglipProcessor
from siglip.custom_siglip import CustomSiglipModel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from transformers import logging as transformer_logging
# Ignore transformer warning
transformer_logging.set_verbosity_error()

import warnings
# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import torchvision.models as models
import torch.nn as nn

def main(args):
    utils.seed_all(args.seed)

    assert args.train_mode in ["whole", "text", "image", "adapter"]
    if args.eval_only:  # 测试阶段
        if args.dyn_moe:  # moe adapters++
            if args.model == "Siglip-B-14-224":
                text_expert_num, image_expert_num = _get_experts_and_router_num_siglip(args.load, args)
                args.text_expert_num_list = text_expert_num
                args.image_expert_num_list = image_expert_num
                # TODO: 模型加载的部分应该封装为函数
                print(f"build pre-trained model config: {args.model}")
                hf_config = AutoConfig.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
                model = CustomSiglipModel.from_pretrained(
                    pretrained_model_name_or_path=args.load,
                    args=args,
                    config=hf_config,
                    ignore_mismatched_sizes=True,
                    )
                model = model.cuda()
                model.eval()
                processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
                image_zise = [processor.image_processor.size["height"], processor.image_processor.size["width"]]
                val_preprocess = build_siglip_transform(image_size=image_zise, is_train=False)
                print("siglip model load!")
            else:
                text_expert_num, image_expert_num = _get_experts_and_router_num(args.load, args)
                args.text_expert_num_list = text_expert_num
                args.image_expert_num_list = image_expert_num
                model, _, val_preprocess = clip.load(args.model, jit=False, args=args)
                print("clip model load!")
        else:
            if args.model == "Siglip-B-14-224":
                print(f"build pre-trained model config: {args.model}")
                hf_config = AutoConfig.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
                model = CustomSiglipModel.from_pretrained(
                    pretrained_model_name_or_path=args.load,
                    args=args,
                    config=hf_config,
                    ignore_mismatched_sizes=True,
                    )
                model = model.cuda()
                model.eval()
                processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
                image_zise = [processor.image_processor.size["height"], processor.image_processor.size["width"]]
                val_preprocess = build_siglip_transform(image_size=image_zise, is_train=False)
                print("siglip model load!") 
            else:
                model, _, val_preprocess = clip.load(args.model, jit=False, args=args)
                print("clip model load!")
        if args.load:  # 加载模型
            if args.model == "Siglip-B-14-224":
                pass
            else:
                utils.torch_load(model, args.load)
                print("train model load!")
        if args.dyn_moe:
            if args.model == "Siglip-B-14-224":
                dyn_evaluate_siglip(model, args, val_preprocess)
            else:
                dyn_evaluate(model, args, val_preprocess)
        else:
            if args.load_autochooser and args.autorouter == True:
                if args.val_preprocess_chooser != None:
                    _, _, val_preprocess_chooser = clip.load("ViT-B/16", jit=False, args=args)
                pretrained_alexnet = models.alexnet(pretrained=True)
                feature_extractor = Alexnet_FE(pretrained_alexnet).cuda()
                Autoencoder_list = nn.ModuleList()
                for i in range(args.task_num + 1):  # more for zero-shot chosen  / few or full shot share the code
                    model_autoencoder = Autoencoder(input_dims=256 * 13 * 13, code_dims=100)
                    Autoencoder_list.append(model_autoencoder)
                utils.torch_load(Autoencoder_list, args.load_autochooser)
                print("autochooser load!")
                Autoencoder_list = Autoencoder_list.cuda()
            elif args.save:  # None
                checkpoint_pth = os.path.join(
                    args.save, f"clip_zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_save(checkpoint_pth, model)
            else:  # no DDAS, no Dyn_MoE
                feature_extractor, Autoencoder_list = None, None
            if args.val_preprocess_chooser != None:
                evaluate(model, feature_extractor, Autoencoder_list, args, val_preprocess, val_preprocess_chooser)
            else:
                evaluate(model, feature_extractor, Autoencoder_list, args, val_preprocess)

    # train
    else:
        if args.train_chooser:  # train DDAS
            if args.few_shot > 0:
                print('----------------------train few-shot chooser----------------------')
                chooser_of_few_shot = few_shot_AutoEncoder(args)  # few shot chooser
            else:
                print('----------------------train full-shot chooser----------------------')
                chooser = AutoEncoder(args)
        elif args.dyn_moe:  # MoE Adapters++
            if args.model == "Siglip-B-14-224":
                finetune_dyn_siglip(args)
            else:
                finetune_dyn(args)
        else:  # train MoE Adapters(routers & experts)
            # print('----------------------finetune model----------------------')
            model = finetune(args)


def _get_experts_and_router_num(model_path, args):
    # 加载保存的模型权重
    checkpoint = torch.load(model_path)

    # 提取state_dict
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    text_expert_counts = []
    image_expert_counts = []

    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i]:
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
        if args.use_dyn_moe_layer_list_visual[i]:
            image_adaptmlp_key = f"visual.transformer.resblocks.{i}.activated_experts_num"
            image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key]
            current_image_experts_num = state_dict[image_matching_keys[0]][0].item()
            image_expert_counts.append(int(current_image_experts_num))
            if current_image_experts_num == [1.0]:
                image_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        else:
            image_expert_counts.append(0)

    return text_expert_counts, image_expert_counts

def _get_experts_and_router_num_siglip(model_path, args):
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
        if args.use_dyn_moe_layer_list_text[i]:
            # 构造需要检查的关键字
            text_adaptmlp_key = f"text_model.encoder.layers.{i}.activated_experts_num"
            # 提取匹配的参数名称
            text_matching_keys = [key for key in state_dict.keys() if
                                  text_adaptmlp_key in key and "vision" not in key]
            current_text_experts_num = state_dict[text_matching_keys[0]][0].item()
            text_expert_counts.append(int(current_text_experts_num))
            if current_text_experts_num == [1.0]:
                text_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        else:
            text_expert_counts.append(0)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            image_adaptmlp_key = f"vision_model.encoder.layers.{i}.activated_experts_num"
            image_matching_keys = [key for key in state_dict.keys() if image_adaptmlp_key in key]
            current_image_experts_num = state_dict[image_matching_keys[0]][0].item()
            image_expert_counts.append(int(current_image_experts_num))
            if current_image_experts_num == [1.0]:
                image_expert_counts.append(args.init_expert_num + 2)  # 如果没有匹配的参数，填入专家数+2
        else:
            image_expert_counts.append(0)

    return text_expert_counts, image_expert_counts

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
