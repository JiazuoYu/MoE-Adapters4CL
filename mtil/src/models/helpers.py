import copy
import os

import clip.clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import datasets, templates, utils

def batch(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_datasets_text(ds, args):
    texts = []
    for d in ds:
        ref_sentences_cls = getattr(datasets, d)
        ref_sentences = ref_sentences_cls(
            None,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        ref_template = ref_sentences.template
        ref_texts = [ref_template(x) for x in ref_sentences.classnames]
        texts.extend(ref_texts)
    ret = clip.tokenize(texts).cuda()
    return ret

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

def moving_avg(model_0, model_1, alpha=0.999):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)

def l2_loss(model, model_ref):
    loss = 0.0
    for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
        loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
    return loss


def virtual_vocab(length=10, n_class=1000):
    voc_len = len(clip._tokenizer.encoder)
    texts = torch.randint(0, voc_len, (n_class, length))
    start = torch.full((n_class, 1), clip._tokenizer.encoder["<start_of_text>"])
    end = torch.full((n_class, 1), clip._tokenizer.encoder["<end_of_text>"])
    zeros = torch.zeros((n_class, 75 - length), dtype=torch.long)

    texts = torch.cat([start, texts, end, zeros], dim=1)
    return texts

def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss

def paired_loss_new(old_pred, old_true):
    T = 2
    pred_soft = F.softmax(old_pred[:, : old_true.shape[0]] / T, dim=1)
    true_soft = F.softmax(old_true[:, : old_true.shape[0]] / T, dim=1)
    loss_old = true_soft.mul(-1 * torch.log(pred_soft))
    loss_old = loss_old.sum(1)
    loss_old = loss_old.mean() * T * T
    return loss_old