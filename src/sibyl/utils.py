import os
import pickle
import numpy as np
import torch

# data find + save + load

def pkl_save(file, path):
    base_path = os.path.dirname(path)
    os.makedirs(base_path, exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def pkl_load(path):
    with open(path, 'rb') as handle:
        file = pickle.load(handle)
    return file

def npy_save(path, file):
    base_path = os.path.dirname(path)
    os.makedirs(base_path, exist_ok=True)
    np.save(path, file)
        
def npy_load(path):
    return np.load(path, allow_pickle=True)

def parse_path_list(path_str, default_path, file_extension='.npy'):
    path_list = []
    input_split = [default_path] if path_str == '' else path_str.split(',')

    for path in input_split:
        if os.path.isfile(path) and path.endswith(file_extension):
            path_list.append(path)
        elif os.path.isdir(path):
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    sub_path = os.path.join(subdir, file)
                    if os.path.isfile(sub_path) and sub_path.endswith(file_extension):
                        path_list.append(sub_path)
        else:
            raise FileNotFoundError('[{}] not exists.'.format(path))

    return path_list

# evaluation metrics

def get_acc_at_k(y_true, y_pred, k=2):
    y_true = torch.tensor(y_true) if type(y_true) != torch.Tensor else y_true
    y_pred = torch.tensor(y_pred) if type(y_pred) != torch.Tensor else y_pred
    total = len(y_true)
    y_weights, y_idx = torch.topk(y_true, k=k, dim=-1)
    out_weights, out_idx = torch.topk(y_pred, k=k, dim=-1)
    correct = torch.sum(torch.eq(y_idx, out_idx) * y_weights)
    acc = correct / total
    return acc.item()

# losses

def CEwST_loss(logits, target, reduction='mean'):
    """
    Cross Entropy with Soft Target (CEwST) Loss
    :param logits: (batch, *)
    :param target: (batch, *) same shape as logits, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(logits.view(logits.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

# misc

def find_max_list(lists):
    list_len = [len(l) for l in lists]
    return max(list_len)

def sample_Xy(text, label, num_sample=1):
    idx = np.random.randint(0, len(text), num_sample)
    return list(np.array(text)[idx]), list(np.array(label)[idx])    

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))