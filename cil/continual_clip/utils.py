
import os
import json
import yaml

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

import pickle
import random

import numpy as np

from clip.tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]


def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_class_names(classes_names, class_ids_per_task):
    return [classes_names[class_id] for class_id in class_ids_per_task]


def get_dataset_class_names(workdir, dataset_name, long=False):
    with open(os.path.join(workdir, "dataset_reqs", f"{dataset_name}_classes.txt"), "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    split_path = path.split("/")
    workdir_idx = split_path.index("cil")
    return "/".join(split_path[:workdir_idx+1])

###########################
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


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": classifier.state_dict()}, save_path)
    print("Checkpoint saved to", save_path)

    # with open(save_path, 'wb') as f:
    #     pickle.dump(classifier.cpu(), f)


def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    missing_keys, unexpected_keys = classifier.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    print("Checkpoint loaded from", save_path)
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)

    if device is not None:
        classifier = classifier.to(device)
    return classifier


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def batch(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def merge_we(model_0, model_1, sma_count):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
    return model_1

def wise_we(model_0, model_1, sma_count, model_n, alpha=0.95):
    for param_q, param_k, param_n in zip(model_0.parameters(), model_1.parameters(), model_n.parameters()):
        param_k.data = (
                        (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
                    ) * alpha + param_n.data * (1-alpha)
    return model_1

def merge_we_router(model_0, model_1, sma_count):
    for param_q, param_k, name_q, name_k in zip(model_0.parameters(), model_1.parameters(), model_0.named_parameters(), model_1.named_parameters()):
        if "router" in name_k[0] or "noise" in name_k[0]:
            param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
            # print('111', name_k[0], name_q[0])
    return model_1

def moving_avg(model_0, model_1, alpha=0.999):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)


def l2_loss(model, model_ref):
    loss = 0.0
    for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
        loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
    return loss


def virtual_vocab(length=10, n_class=1000):
    voc_len = len(_tokenizer.encoder)
    # breakpoint()
    texts = torch.randint(0, voc_len, (n_class, length))
    start = torch.full((n_class, 1), _tokenizer.encoder["<start_of_text>"])
    end = torch.full((n_class, 1), _tokenizer.encoder["<end_of_text>"])
    zeros = torch.zeros((n_class, 75 - length), dtype=torch.long)

    texts = torch.cat([start, texts, end, zeros], dim=1)
    return texts
    
def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss



