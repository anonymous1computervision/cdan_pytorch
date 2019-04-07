import torch
import numpy as np
from dixitool.pytorch.optim import functional as optimizerF
from dixitool.pytorch.module import functional as moduleF
import torchvision.utils 
import torch.optim as optim

def train():

    len_src = len(src_trainloader)
    len_tgt = len(tgt_trainloader)
    num_iter = max(len_src, len_tgt)

    for batch_idx in range(num_iter):
        if batch_idx % len_src == 0:
            iter_src = iter(src_trainloader)
        if batch_idx % len_tgt == 0:
            iter_tgt = iter(tgt_trainloader)

        src_inputs, src_labels = iter_src.next()
        tgt_inputs, tgt_labels = iter_tgt.next()

        #########
        #Train
        #########
        

def test():

    return 0