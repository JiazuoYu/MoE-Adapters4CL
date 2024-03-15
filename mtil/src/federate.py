import copy
import os
from random import random

import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, evaluate_fc, evaluate_wise_ft, finetune, finetune_fc
from .models.modeling import create_image_classifier



def main(args):
    model_basic, _, val_preprocess = clip.load(args.model, jit=False)
    utils.torch_load(model_basic, args.load_federate[0])
    model_basic.cuda()
    
    if len(args.load_federate) > 1:
        for model_ckp in args.load_federate[1:]:
            model_new, _, val_preprocess = clip.load(args.model, jit=False)
            utils.torch_load(model_new, model_ckp)
            model_new.cuda()
            for i, (param_q, param_k) in enumerate(zip(model_basic.parameters(), model_new.parameters())):
                param_q.data = param_k.data + param_q.data 
        for param_q in model_basic.parameters():
            param_q.data /= len(args.load_federate)
    assert args.train_mode in ["whole", "text", "image"]
    evaluate(model_basic, args, val_preprocess)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)