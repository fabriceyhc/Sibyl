import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import itertools

from utils import *

def rand_bbox(size, lam):
    W, H = size[2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix2(
    batch, 
    alpha, 
    target_pairs, 
    target_prob, 
    resize_prob, 
    num_classes):
    """
    Applies the cutmix algorithm to a pytorch batch.
    Optionally targets specific classes for mixing.

    Parameters
    ----------
    batch : tuple (torch.Tensor, torch.Tensor)
        The input data and the targets, which will be 
        one-hot-encoded if they aren't already
    alpha : float
        Shape parameter used for the beta distribution 
        sample. Used for both a and b. 
    target_pairs : list[tuple(int, int)]
        A list of tuple pairs that represent a mapping
        between a source_class and target_class. The 
        target_class gets cut into the source_class. 
        For all unspecified class pairings, a random 
        pairing will be made
    target_prob : float [0,1]
        The probability of applying the targeted cutmix
        for a particular pairing
    resize_prob : float [0,1]
        Instead of copying and pasting a subset of the
        pixels into another image, copy the whole image
        and resize it so that it fits inside the bbox
    num_classes : int
        The number of classes in the dataset

    Returns
    -------
    tuple (torch.Tensor, torch.Tensor)
        A tuple containing the data and the one hot encoded targets 
    """

    # unpack batch
    data, targets = batch
    batch_size, C, H, W = data.size()

    # create placeholder for distributional labels
    ohe_targets = torch.zeros((batch_size, num_classes)).float() # F.one_hot(targets, num_classes).float()

    # draw a new lambda value to weight the two classes
    lam = np.random.beta(alpha, alpha)

    # randomly select bounding box for insert
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    # resizing preliminaries
    resize = torch.rand(1) < resize_prob
    if resize:
        new_H = bbx2 - bbx1
        new_W = bby2 - bby1
        if new_H == 0 or new_W == 0:
            resize = False

    # track indices for targeted cutmix to exclude later
    idx = [x for x in range(batch_size)]
    ex_idx = []

    # cutmix targeted pairings
    for pair in target_pairs:

        # skip targeted cutmix target_prob percent of the time
        use_target_cutmix = torch.rand(1) < target_prob
        if not use_target_cutmix:
            continue

        # unpack source and target pairs
        source_class, target_class = pair

        # find indices of both source and target 
        s_idx = find_value_idx(targets, source_class)
        t_idx = find_value_idx(targets, target_class)

        # if none of the source or target classes are in this batch, skip it
        if len(s_idx) == 0 or len(t_idx) == 0:
            continue

        # enforce tensor size by drawing from a distribution
        weights = torch.ones_like(t_idx).float()
        tt_idx = torch.multinomial(weights, len(s_idx), replacement=True)
        t_idx = t_idx[tt_idx]

        # cutmix the target imgs into the source imgs
        if resize:
            data[s_idx, :, bbx1:bbx2, bby1:bby2] = F.interpolate(data[t_idx], size=(new_H, new_W))
        else:
            data[s_idx, :, bbx1:bbx2, bby1:bby2] = data[t_idx, :, bbx1:bbx2, bby1:bby2]
        
        # reset lambda to be proportionate to the bbox size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (np.prod(data.size()[2:])))
        
        # distribute the label proportional to bbox size
        ohe_targets[s_idx, source_class] += lam
        ohe_targets[s_idx, target_class] += (1-lam)

        # exclude targeted indices for random pairings
        ex_idx.append(s_idx.tolist())

    # shuffle remaining indices to randomly pair untargeted classes
    ex_idx = list(itertools.chain(*ex_idx))
    s_idx = [x for x in idx if x not in ex_idx]
    t_idx = random.sample(s_idx, len(s_idx))

    if resize:
        data[s_idx, :, bbx1:bbx2, bby1:bby2] = F.interpolate(data[t_idx], size=(new_H, new_W))
    else:
        data[s_idx, :, bbx1:bbx2, bby1:bby2] = data[t_idx, :, bbx1:bbx2, bby1:bby2]  

    ohe_targets[s_idx, targets[s_idx]] += lam
    ohe_targets[s_idx, targets[t_idx]] += (1-lam)

    return data, ohe_targets

class CutMix2Collator:
    def __init__(self, 
        alpha=1.0,  
        target_pairs=[], 
        target_prob=1.0, 
        resize_prob=0.0, 
        num_classes=10):

        self.alpha = alpha
        self.target_pairs = target_pairs
        self.target_prob = target_prob
        self.resize_prob = resize_prob
        self.num_classes = num_classes

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix2(
            batch, 
            self.alpha, 
            self.target_pairs,   
            self.target_prob,
            self.resize_prob,
            self.num_classes
        )
        return batch