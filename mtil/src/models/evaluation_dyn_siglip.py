# --------------------------------------------------------
# Evaluation code of MoE-Adapters++ for a single task, based on SIGLIP
# --------------------------------------------------------
import siglip

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
import os

mtil_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(mtil_path)

from siglip.encoder_layer.moe_layer import get_eval_zero_shot, get_val_task_id_visual
from transformers import AutoProcessor

# from .finetune import get_experts_and_router_num
from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize
import torch.nn.functional as F
from .AutoEncoder import encoder_criterion


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # print('pred',pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model, args):
    processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    # for task_id in range(args.task_num):
    for task_id in range(-1, args.task_num):
        zeroshot_weights_i = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = processor(text=texts,padding="max_length")["input_ids"].cuda()
            # texts = clip.tokenize(texts).cuda()  # tokenize
            if args.non_text == True:
                # class_embeddings = model.encode_text(texts, -1)  # embed with text encoder
                class_embeddings = model.get_text_features(texts)  # embed with text encoder
            else:
                # class_embeddings = model.encode_text(texts, task_id)  # embed with text encoder
                class_embeddings = model.get_text_features(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights_i.append(class_embedding)
        zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
        zeroshot_weights.append(zeroshot_weights_i)
    return zeroshot_weights


@torch.no_grad()
def dyn_zeroshot_eval(model, loader, zeroshot_weights, args):
    # initial TensorBoard SummaryWriter 
    writer = None
    if args.log_dir is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader, ncols=50)):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()
        # predict
        image_features = model.get_image_features(images)
        # image_features, image_features_original = model.encode_image(images)
        # normalize
        image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        task_id = get_val_task_id_visual(-1)
        zero_shot_flag = get_eval_zero_shot()
        if zero_shot_flag:
            logits = 100.0 * image_features @ zeroshot_weights[0]
        else:
            logits = 100.0 * image_features @ zeroshot_weights[task_id + 1]
        
        # if args.augmented_zero_shot and zero_shot_flag: # 在siglip version中已经废弃
        #     image_features_original /= image_features_original.norm(dim=-1, keepdim=True)
        #     logits_original = 100.0 * image_features_original @ zeroshot_weights[0]
            
        #     max_logits_original, _ = torch.max(logits_original, dim=1)
        #     max_logits, _ = torch.max(logits, dim=1)

        #     # Create a mask to find out which lines need to be replaced
        #     mask = max_logits_original > max_logits
        #     logits[mask] = logits_original[mask]    

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)
        if writer is not None:
            eval_loss_to_tensorboard(model, writer, i, args,)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    return top1, top5


def dyn_eval_single_dataset(image_classifier, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()
    print(f"Get text embeddings of {args.eval_datasets}...")
    # get_expert_freq_list(model)
    # print('dataset',dataset.classnames)  # vocabulary list
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model, args
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    print(f"Text embeddings processed!")
    top1, top5 = dyn_zeroshot_eval(model, dataloader, zeroshot_weights, args)

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    get_eval_acc(model, args)
    if args.force_val_task_id is not None:
        print(f"Force val_task_id: {args.force_val_task_id}")
    # print(f"LR_CE:{args.lr}")
    # print(f"LR_AE:{args.lr_rd}")
    # print(f"ZS_threshold:{args.zero_shot_threshold_image}")


def dyn_evaluate_siglip(image_classifier, args, val_preprocess):

    if args.eval_datasets is None:
        return
    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)  # Caltech101
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        dyn_eval_single_dataset(image_classifier, dataset, args)


def calculate_list_probability(eval_list, eval_id):
    if not eval_list:
        return 0  
    count_one = eval_list.count(eval_id) 
    total_count = len(eval_list)
    probability = count_one / total_count 
    return probability


def get_eval_acc(model, args):
    for i in range(len(args.use_dyn_moe_layer_list_text)):
        if args.track_val_task_id[i]:
            if args.use_dyn_moe_layer_list_text[i]:
                text_eval_acc_list = model.text_model.encoder.layers[i].eval_acc_list
                # text
                text_zero_shot_rate = calculate_list_probability(text_eval_acc_list, -1)
                text_right_rate = calculate_list_probability(text_eval_acc_list, args.eval_acc_task_id)  # * 100
                if model.text_model.encoder.layers[i].reconginition_layer:
                    if args.force_val_task_id is None:
                        print(f"text_zero_shot_rate: {text_zero_shot_rate:.2f}")
                        print(f"text_right_rate: {text_right_rate:.2f}")
            if args.use_dyn_moe_layer_list_visual[i]:
                visual_eval_acc_list = model.vision_model.encoder.layers[i].eval_acc_list
                # visual
                visual_zero_shot_rate = calculate_list_probability(visual_eval_acc_list, -1)
                visual_right_rate = calculate_list_probability(visual_eval_acc_list, args.eval_acc_task_id)  # * 100
                if model.vision_model.encoder.layers[i].reconginition_layer:
                    print(f"visual_zero_shot_rate: {visual_zero_shot_rate:.2f}")
                    print(f"visual_right_rate: {visual_right_rate:.2f}")
        if args.track_val_discrepancy[i]:
            if args.use_dyn_moe_layer_list_text[i] and model.text_model.encoder.layers[i].reconginition_layer:
                text_eval_discrepancy_list = model.text_model.encoder.layers[i].eval_discrepancy_list
                print(f"Max discrepancy: {max(text_eval_discrepancy_list):.4f}")
                print(f"Min discrepancy: {min(text_eval_discrepancy_list):.4f}")
            if args.use_dyn_moe_layer_list_visual[i] and model.vision_model.encoder.layers[i].reconginition_layer:
                visual_eval_discrepancy_list = model.vision_model.encoder.layers[i].eval_discrepancy_list
                if args.force_val_task_id is None and args.force_zero_shot is False:
                    print(f"Max discrepancy: {max(visual_eval_discrepancy_list):.4f}")
                    print(f"Min discrepancy: {min(visual_eval_discrepancy_list):.4f}")


def get_expert_freq_list(model, args):
    for i in range(args.text_layer):
        text_expert_freq_list = model.text_model.encoder.layers[i].expert_activate_freq_list
        print(f"Text layer{i}:")
        for idx, param in enumerate(text_expert_freq_list):
            formatted_values = ' '.join([f"{x:.2f}" for x in param.tolist()])
            print(formatted_values)
    for i in range(args.vision_layer):
        visual_expert_freq_list = model.vision_model.encoder.layers[i].expert_activate_freq_list
        print(f"Visual layer{i}:")
        for idx, param in enumerate(visual_expert_freq_list):
            formatted_values = ' '.join([f"{x:.2f}" for x in param.tolist()])
            print(formatted_values)


def eval_loss_to_tensorboard(model, writer, idx, args,):
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i]:
            text_mse_loss_list = [x for x in model.text_model.encoder.layers[i].eval_mse_list if x != 0]
            # Recording Losses to TensorBoard
            text_loss_dict = {f'Eval Loss of RD{i}': loss for i, loss in enumerate(text_mse_loss_list)}
            writer.add_scalars(f'Eval_Text_losses_of_layer{i}', text_loss_dict, idx)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            visual_mse_loss_list = [x for x in model.vision_model.encoder.layers[i].eval_mse_list if x != 0]
            # Recording Losses to TensorBoard
            visual_loss_dict = {f'Eval Loss of RD{i}': loss for i, loss in enumerate(visual_mse_loss_list)}
            writer.add_scalars(f'Eval_Visual_losses_of_layer{i}', visual_loss_dict, idx)
