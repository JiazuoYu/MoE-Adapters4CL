# --------------------------------------------------------
# Implementation of the loss for MoE-Adapters++
# including CE_loss for classification
# and MSE_loss for Auto-Encoder 
# --------------------------------------------------------
import torch.nn.functional as F
from torch import nn


def MSE_loss(x, reconstructed_x):
    """
    The reconstruction loss on all the features fed to Adapters.
    See paper: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning
    """
    rd_loss = nn.MSELoss()
    return rd_loss(reconstructed_x, x)


def total_loss(ce_loss,
               mse_loss,
               args):
    total_loss = args.ce_weight * ce_loss
    total_loss += args.mse_weight * mse_loss
    return total_loss
