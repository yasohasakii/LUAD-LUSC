import torch
import torch.nn as nn
import torch.nn.functional as F
import network
import losses
import numpy as np
from fastai.optimizer import OptimWrapper
from fastai.optimizer import SGD

def train(n_epoch = 50,
          ):
    model = network.MILModel(n_feats=128,n_out=1)


