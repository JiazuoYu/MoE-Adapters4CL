import clip

import torch
from tqdm import tqdm

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
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    # for task_id in range(args.task_num):
    for task_id in range(-1, args.task_num):
        zeroshot_weights_i = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            if args.non_text == True:
                class_embeddings = model.encode_text(texts, -1)  # embed with text encoder
            else:
                class_embeddings = model.encode_text(texts,task_id)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights_i.append(class_embedding)
        zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
        zeroshot_weights.append(zeroshot_weights_i)
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model,feature_extractor,Autoencoder_list, loader, zeroshot_weights, args):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict batch image domain:
        input_to_ae = feature_extractor(images)
        input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
        input_to_ae = input_to_ae
        input_to_ae = F.sigmoid(input_to_ae)  # GT

        model_autoencoder = Autoencoder_list[0]
        outputs = model_autoencoder(input_to_ae)
        best_l = encoder_criterion(outputs, input_to_ae)
        best_router = 0
        for i in range(1, 12):
            outputs = Autoencoder_list[i](input_to_ae)
            new_l = encoder_criterion(outputs, input_to_ae)
            if new_l < best_l:
                best_l = new_l
                best_router = i
            if best_l > args.threshold:
                best_router = 0
        task_id = best_router - 1
        # predict
        image_features = model.encode_image(images, task_id)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights[task_id+1]
        # [zeroshot_weights]:-1 to 11
        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier,feature_extractor,Autoencoder_list, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    Autoencoder_list.eval()
    model.eval()
    # print('dataset',dataset.classnames)  # vocabulary list
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model,args
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    top1, top5 = zeroshot_eval(model,feature_extractor,Autoencoder_list, dataloader, zeroshot_weights, args)

    print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")


def evaluate(image_classifier,feature_extractor,Autoencoder_list, args, val_preprocess):
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
        eval_single_dataset(image_classifier,feature_extractor,Autoencoder_list, dataset, args)
