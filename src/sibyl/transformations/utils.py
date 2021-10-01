import numpy as np
import torch

def already_ohe(y):
    if len(y.shape) <= 1:
        return False
    return (y.sum(axis=1)-np.ones(y.shape[0])).sum() == 0

def one_hot_encode(y, nb_classes):
    if isinstance(y, (np.ndarray, torch.Tensor)):
        if len(y.shape) == 1:
            y = np.expand_dims(y, 0)   
    else:
        y = np.expand_dims(np.array(y), 0)
    try:
        res = np.eye(nb_classes)[y.reshape(-1)]
        res = res.reshape(list(y.shape)+[nb_classes]).squeeze()
        return res
    except:
        return y

def soften_label(y, num_classes=None):
    if not num_classes:
        if isinstance(y, int):
            num_classes = max(2, y + 1)
        elif isinstance(y, list):
            num_classes = len(y)
        elif isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                num_classes = len(y)
            else:
                num_classes = y.shape[-1]
    return one_hot_encode(y, num_classes) 

def invert_label(y, soften=False, num_classes=None):
    if not isinstance(y, np.ndarray):
        y = soften_label(y, num_classes)
    y = y[::-1]
    if not soften:
        y = np.argmax(y)
    return y

def interpolate_label(y1, y2, x1=None, x2=None, num_classes=None, y_weights=None):
    if isinstance(y_weights, (list, tuple)):
        mass_y1 = y_weights[0]
        mass_y2 = y_weights[1]
    elif x1 and x2:
        mass_y1 = len(x1) / (len(x1) + len(x2)) 
        mass_y2 = 1 - mass_y1
    else:
        mass_y1 = 1
        mass_y2 = 1    
    y1 = soften_label(y1, num_classes) * mass_y1
    y2 = soften_label(y2, num_classes) * mass_y2
    return (y1 + y2) / (y1 + y2).sum()

def weight_label(y, y_weights=None):
    if not y_weights:
        y_weights = np.ones_like(y)
    y = y * np.array(y_weights)
    return y / y.sum()

def smooth_label(y, factor=0.1):
    if not isinstance(y, np.ndarray):
        y = soften_label(y)
    y = y * (1. - factor)
    y = y + (factor / y.shape[-1])
    return y

# labels1 = [0, 1]
# labels2 = np.array([0, 1])

# hard_label_int   = 3
# hard_labels_list = [0,1,2,3]
# hard_labels_arr  = np.array([0,1,2,3])
# soft_labels_arr  = np.array([0.45035461, 0., 0., 0.54964539])

# print(one_hot_encode(labels1, 2))
# print(one_hot_encode(labels2, 2))

# print(one_hot_encode(hard_label_int, 4))
# print(one_hot_encode(hard_labels_list, 4))
# print(one_hot_encode(hard_labels_arr, 4))
# print(one_hot_encode(soft_labels_arr, 4))