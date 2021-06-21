import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import numpy as np
import random
import itertools
import math

from utils import *

def tile(
    batch, 
    num_tiles, 
    target_pairs, 
    target_prob, 
    num_classes):
    """
    Applies the tile algorithm to a pytorch batch.
    Optionally targets specific classes for mixing.

    Parameters
    ----------
    batch : tuple (torch.Tensor, torch.Tensor)
        The input data and the targets, which will be 
        one-hot-encoded if they aren't already
    num_tiles : int
        The number of images to be tiled together
    num_classes : int
        The number of classes in the dataset

    Returns
    -------
    tuple (torch.Tensor, torch.Tensor)
        A tuple containing the data and the distributional targets 
    """

    # unpack batch
    data, targets = batch
    batch_size, C, H, W = data.size()

    use_tile = torch.rand(1) < target_prob
    has_targets = False

    if use_tile and target_pairs:
        s_lbls, t_lbls  = target_pairs
        values, indices = torch.sort(targets)
        all_data = []
        for s in s_lbls:
            s_idx = indices[values == s]
            if len(s_idx) == 0:
                continue 
            t_data = []
            for i, t in enumerate(t_lbls[s]):   
                t_idx = indices[values == t]
                if len(t_idx) == 0:
                    num = len(s_idx)
                    t_idx = torch.randperm(batch_size)[:num]
                weights = torch.ones_like(t_idx).float()
                tt_idx = torch.multinomial(weights, len(s_idx), replacement=True)
                t_idx = t_idx[tt_idx]
                t_data.append(t_idx)
            all_data.append(torch.stack((s_idx, *t_data)).T)    
        t_idx = torch.cat(all_data).T
        has_targets = True

    # randomly permute indices and stack them 
    X = []
    y = []
    for i in range(num_tiles):
        if use_tile and has_targets:
            idx = t_idx[i]
        else:
            idx = torch.randperm(batch_size)
        X.append(data[idx])
        y.append(targets[idx])
    X = torch.stack(X, dim=1)
    y = torch.stack(y, dim=1)

    # turn each set of images into a gridded image
    nrow = math.ceil(math.sqrt(num_tiles))

    new_data = []
    for i in range(batch_size):
        X_grid = make_grid(tensor=X[i],nrow=nrow,padding=0)
        new_data.append(X_grid)
    new_data = torch.stack(new_data, dim=0)[:,:C,:,:]

    # set class weighting and generate new distributed target label
    lam = 1. / num_tiles  
    new_targets = F.one_hot(y, num_classes).sum(dim=1) * lam

    return new_data, new_targets

class TileCollator:
    def __init__(self, 
        num_tiles, 
        target_pairs,
        target_prob, 
        num_classes):

        self.num_tiles = num_tiles
        self.target_pairs = target_pairs
        self.target_prob = target_prob
        self.num_classes = num_classes

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = tile(
            batch, 
            self.num_tiles, 
            self.target_pairs, 
            self.target_prob, 
            self.num_classes
        )
        return batch