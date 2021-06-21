import numpy as np

def one_hot_encode(y, nb_classes):
    if isinstance(y, np.ndarray):
        if len(y.shape) == 1:
            y = np.expand_dims(y, 0)
    else:
        y = np.expand_dims(np.array(y), 0)
    if y.shape[-1] == nb_classes:
        return y
    res = np.eye(nb_classes)[y.reshape(-1)]
    return res.reshape(list(y.shape)+[nb_classes]).squeeze()

def soften_label(y, num_classes=None):
    if not num_classes:
        if isinstance(y, int):
            num_classes = max(2, y)
        elif isinstance(y, list):
            num_classes = len(y)
        elif isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                num_classes = len(y)
            else:
                num_classes = y.shape[-1]
    return one_hot_encode(y, num_classes) 

def invert_label(y, soften=False, num_classes=None):
    if soften:
        y = soften_label(y, num_classes)
    if isinstance(y, np.ndarray):
        return (1 - y) / (1 - y).sum()
    else:
        return int(not y)

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