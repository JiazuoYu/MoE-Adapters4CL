import clip

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from .finetune import get_experts_and_router_num
from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize
import torch.nn.functional as F
from .AutoEncoder import encoder_criterion
from transformers import AutoProcessor


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
    if args.model == "Siglip-B-14-224":
        processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    # for task_id in range(args.task_num):
    for task_id in range(-1, args.task_num):
        zeroshot_weights_i = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            if args.model == "Siglip-B-14-224":
                # 这里是Siglip-B-14-224的处理
                texts = processor(text=texts,padding="max_length")["input_ids"].cuda()
            else:
                texts = clip.tokenize(texts).cuda()  # tokenize
            if args.non_text == True:
                if args.model == "Siglip-B-14-224":
                    class_embeddings = model.get_text_features(texts, val_task_id=-1)  # embed with text encoder
                else:
                    class_embeddings = model.encode_text(texts, -1)  # embed with text encoder
            else:
                if args.model == "Siglip-B-14-224":
                    class_embeddings = model.get_text_features(texts, val_task_id=task_id)  # embed with text encoder
                else:
                    class_embeddings = model.encode_text(texts, task_id)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights_i.append(class_embedding)
        zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
        zeroshot_weights.append(zeroshot_weights_i)
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, feature_extractor, Autoencoder_list, loader, zeroshot_weights, args, loader_chooser=None):
    top1, top5, n = 0.0, 0.0, 0.0
    if loader_chooser != None:
        for i, (data, data_chooser) in enumerate(tqdm(zip(loader, loader_chooser), total=len(loader), ncols=50)):
            data = maybe_dictionarize(data)
            data_chooser = maybe_dictionarize(data_chooser)
            images_chooser = data_chooser["images"].cuda()
            images = data["images"].cuda()
            target = data["labels"].cuda()

            if args.force_val_task_id is not None:
                task_id = args.force_val_task_id
            else:
                # predict batch image domain:
                # print(images_chooser.shape)
                input_to_ae = feature_extractor(images_chooser)
                input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
                input_to_ae = input_to_ae
                input_to_ae = F.sigmoid(input_to_ae)  # GT

                model_autoencoder = Autoencoder_list[0]
                # print(input_to_ae.shape)
                outputs = model_autoencoder(input_to_ae)
                # 这里是DDAS判断阈值,首先确认当前效果最好的loss
                best_l = encoder_criterion(outputs, input_to_ae)
                best_router = 0
                for i in range(1, len(Autoencoder_list)):
                    # 分别检查所有任务的auto encoder
                    outputs = Autoencoder_list[i](input_to_ae)
                    new_l = encoder_criterion(outputs, input_to_ae)  # 论文中的dt
                    if new_l < best_l:
                        best_l = new_l
                        best_router = i
                    if best_l > args.threshold:
                        best_router = 0
                task_id = best_router - 1
                # task_id = best_router
                print('task id   *  ', task_id)
            # predict
            if args.model == "Siglip-B-14-224":
                # print("task_id", task_id)
                image_features = model.get_image_features(images, val_task_id=task_id)
                image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            else:
                image_features, _ = model.encode_image(images, task_id)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights[task_id + 1]
            # [zeroshot_weights]:-1 to 11
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        return top1, top5
    else: 
        for i, data in enumerate(tqdm(loader, ncols=50)):

            data = maybe_dictionarize(data)
            images = data["images"].cuda()
            target = data["labels"].cuda()

            if args.force_val_task_id is not None:
                task_id = args.force_val_task_id
            else:
                # predict batch image domain:
                # print(images.shape)
                input_to_ae = feature_extractor(images)
                input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
                input_to_ae = input_to_ae
                input_to_ae = F.sigmoid(input_to_ae)  # GT

                model_autoencoder = Autoencoder_list[0]
                # print(input_to_ae.shape)
                outputs = model_autoencoder(input_to_ae)
                # 这里是DDAS判断阈值,首先确认当前效果最好的loss
                best_l = encoder_criterion(outputs, input_to_ae)
                best_router = 0
                for i in range(1, len(Autoencoder_list)):
                    # 分别检查所有任务的auto encoder
                    outputs = Autoencoder_list[i](input_to_ae)
                    new_l = encoder_criterion(outputs, input_to_ae)  # 论文中的dt
                    if new_l < best_l:
                        best_l = new_l
                        best_router = i
                    if best_l > args.threshold:
                        best_router = 0
                task_id = best_router - 1
                # task_id = best_router
                print('task id   *  ', task_id)
            # predict
            if args.model == "Siglip-B-14-224":
                # print("task_id", task_id)
                image_features = model.get_image_features(images, val_task_id=task_id)
                image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            else:
                image_features, _ = model.encode_image(images, task_id)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights[task_id + 1]
            # [zeroshot_weights]:-1 to 11
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        return top1, top5


def eval_single_dataset(image_classifier, feature_extractor, Autoencoder_list, dataset, args, dataset_chooser=None):
    model = image_classifier
    input_key = "images"
    image_enc = None
    if Autoencoder_list is not None:
        Autoencoder_list.eval()
    model.eval()
    print(f"Get text embeddings of {args.eval_datasets}...")
    # print('dataset',dataset.classnames)  # vocabulary list
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model, args
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    if dataset_chooser !=None:
        dataloader_chooser = get_dataloader(
        dataset_chooser, is_train=False, args=args, image_encoder=image_enc
    )
    if dataset_chooser !=None:
        top1, top5 = zeroshot_eval(model, feature_extractor, Autoencoder_list, dataloader, zeroshot_weights, args, dataloader_chooser)
    else:
        top1, top5 = zeroshot_eval(model, feature_extractor, Autoencoder_list, dataloader, zeroshot_weights, args)

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    # print(f"LR:{args.lr}")


def evaluate(image_classifier, feature_extractor, Autoencoder_list, args, val_preprocess, val_preprocess_chooser=None):
    if args.eval_datasets is None:
        return
    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)  
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        if val_preprocess_chooser != None:
            dataset_chooser = dataset_class(
            val_preprocess_chooser,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
            eval_single_dataset(image_classifier, feature_extractor, Autoencoder_list, dataset, args, dataset_chooser)
        else:
            eval_single_dataset(image_classifier, feature_extractor, Autoencoder_list, dataset, args)


@torch.no_grad()
def dyn_zeroshot_eval(model, loader, zeroshot_weights, args):
    # 初始化 TensorBoard SummaryWriter
    writer = None
    if args.log_dir is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()
        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights[0]
        # [zeroshot_weights]:-1 to 11
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
    print(f"LR:{args.lr}")
    # print(f"exp_threshold_image:{args.expansion_threshold_image}")


def dyn_evaluate(image_classifier, args, val_preprocess):

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
        return 0  # 如果列表为空，返回0
    count_one = eval_list.count(eval_id)  # 计算 -1 的数量
    total_count = len(eval_list)  # 计算列表的总长度
    probability = count_one / total_count  # 计算概率
    return probability


def get_eval_acc(model, args):
    for i in range(args.text_layer):
        if args.track_val_task_id[i]:
            if args.use_dyn_moe_layer_list_text[i]:
                text_eval_acc_list = model.transformer.resblocks[i].eval_acc_list
                # text
                text_zero_shot_rate = calculate_list_probability(text_eval_acc_list, -1)
                text_right_rate = calculate_list_probability(text_eval_acc_list, args.eval_acc_task_id)
                print(
                    f"Layer {i}, text_zero_shot_rate: {text_zero_shot_rate:.2f}, text_right_rate: {text_right_rate:.2f}"
                )
    for i in range(args.vision_layer):
        if args.track_val_task_id[i]:
            if args.use_dyn_moe_layer_list_visual[i]:
                visual_eval_acc_list = model.visual.transformer.resblocks[i].eval_acc_list
                # visual
                visual_zero_shot_rate = calculate_list_probability(visual_eval_acc_list, -1)
                visual_right_rate = calculate_list_probability(visual_eval_acc_list, args.eval_acc_task_id)
                print(
                    f"Layer {i}, visual_zero_shot_rate: {visual_zero_shot_rate:.2f}, visual_right_rate: {visual_right_rate:.2f}"
                )


def get_expert_freq_list(model, args):
    for i in range(args.text_layer):
        text_expert_freq_list = model.transformer.resblocks[i].expert_activate_freq_list
        print(f"Text layer{i}:")
        for idx, param in enumerate(text_expert_freq_list):
            formatted_values = ' '.join([f"{x:.2f}" for x in param.tolist()])
            print(formatted_values)
    for i in range(args.vision_layer):
        visual_expert_freq_list = model.visual.transformer.resblocks[i].expert_activate_freq_list
        print(f"Visual layer{i}:")
        for idx, param in enumerate(visual_expert_freq_list):
            formatted_values = ' '.join([f"{x:.2f}" for x in param.tolist()])
            print(formatted_values)


def eval_loss_to_tensorboard(model, writer, idx, args,):
    for i in range(args.text_layer):
        if args.use_dyn_moe_layer_list_text[i]:
            text_rd_loss_list = [x for x in model.transformer.resblocks[i].eval_rd_list if x != 0]
            # 记录损失到 TensorBoard
            text_loss_dict = {f'Eval Loss of RD{i}': loss for i, loss in enumerate(text_rd_loss_list)}
            writer.add_scalars(f'Eval_Text_losses_of_layer{i}', text_loss_dict, idx)
    for i in range(args.vision_layer):
        if args.use_dyn_moe_layer_list_visual[i]:
            visual_rd_loss_list = [x for x in model.visual.transformer.resblocks[i].eval_rd_list if x != 0]
            # 记录损失到 TensorBoard
            visual_loss_dict = {f'Eval Loss of RD{i}': loss for i, loss in enumerate(visual_rd_loss_list)}
            writer.add_scalars(f'Eval_Visual_losses_of_layer{i}', visual_loss_dict, idx)
