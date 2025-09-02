import torchvision.models as models
import torch.nn as nn
import clip.clip as clip
from torch.utils.data import DataLoader

from .. import datasets, templates, utils
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
from ..datasets.common import get_dataloader, maybe_dictionarize


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=10):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    """
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    # print('lr is ' + str(lr))

    # if (epoch % lr_decay_epoch == 0):
    # print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train_rd(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, val_preprocess = clip.load(args.model, jit=False, args=args)