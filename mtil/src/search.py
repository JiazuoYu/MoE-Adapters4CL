import os

import clip.clip as clip
from tqdm import tqdm
from .args import parse_arguments


import torch
import torch.nn.functional as F
import numpy as np

from . import datasets, templates

# from src.models.utils import cosine_lr

# from .models.eval_baseline import evaluate_baseline

from . import datasets
from .datasets.common import get_dataloader, maybe_dictionarize

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)


    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
    return top1


def evaluate_baseline(image_classifier, args, val_preprocess, eval_dataset):
    if eval_dataset is None:
        return
    for i, dataset_name in enumerate(eval_dataset):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        return eval_single_dataset(image_classifier, dataset, args)

def search(ds, lrs, args):
    for dataset in ds:
        for lr in lrs:
            search_single(dataset, lr, args)
   
def search_single(dat, lr, args): 

    SCORES =  []

    EPOCH = 10

    print("loading CLIP model")
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)

    print("[Training mode] Both Encoders")
    params = model.parameters()

    # prepare dataset
    dataset_class = getattr(datasets, dat)
    dataset = dataset_class(
        train_preprocess, location=args.data_location, batch_size=128
    )
    
    # prepare template
    template = dataset.template

    num_batches = len(dataset.train_loader)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    scheduler = cosine_lr(
        optimizer, lr, args.warmup_length, EPOCH * num_batches
    )

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)

    idx_to_class = {v: k for k, v in dataset.train_dataset.class_to_idx.items()}

    for n_epoch in range(EPOCH):
        model.train()
        for n_step, (images, labels) in enumerate(tqdm(dataset.train_loader)):
            step = n_step + n_epoch * num_batches
            scheduler(step)

            lst = []
            for idx in labels:
                lst.append(idx_to_class[idx.item()])

            # prepare text & image
            text = [template(x) for x in lst]
            text = clip.tokenize(text).cuda()
            images = images.cuda()

            logits_per_image, logits_per_text = model(images, text)
            label = torch.arange(len(images)).cuda()
            loss_i = F.cross_entropy(logits_per_image, label)
            loss_t = F.cross_entropy(logits_per_text, label)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(" ")
        print(f'[EPOCH]: {n_epoch} with [LEARNING_RATE]: {lr}')
        SCORES.append(
            evaluate_baseline(model, args, val_preprocess, [dat])
        )
        model.train()

        # Saving model
        # if n_epoch in [0, 9, 14, 19, 24, 29]:
        #     if args.save is not None:
        #         os.makedirs(args.save, exist_ok=True)
        #         model_path = os.path.join(args.save, f"checkpoint_{n_epoch+1}.pt")
        #         print("Saving model to", model_path)
        #         torch.save(model.state_dict(), model_path)

    if SCORES:
        scores = np.array(SCORES)
        np.save(f"generate_scores/{dat}_with_{lr}.npy", scores)

    # if args.save is not None:
    #     return model_path
    return 


if __name__ == "__main__":

    args = parse_arguments()
    # breakpoint()
    small = ["CIFAR100","SUN397", "EuroSAT"]
    maybe = ["Food", "OxfordPet", "StanfordCars"]
    large  = ["CIFAR100", "EuroSAT", "SUN397"]
    ds, lrs =  small, [1e-5, 5e-6]
    search(ds, lrs, args)