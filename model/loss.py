from traceback import print_tb
import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def ce_loss_rnn(output, target):    
    return F.cross_entropy(output, target, reduction='sum')

def ce_loss(output, target):
    return F.cross_entropy(output, target, reduction='mean')

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def bcewl_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)  
