import os
import pickle
import numpy as np
import torch

from transformers import (
    Trainer,  
    TrainerCallback
)

from sklearn.metrics import accuracy_score
from transformers.trainer_callback import TrainerControl
import pandas as pd

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

# misc

def find_max_list(lists):
    list_len = [len(l) for l in lists]
    return max(list_len)

def sample_Xy(text, label, num_sample=1):
    idx = np.random.randint(0, len(text), num_sample)
    return list(np.array(text)[idx]), list(np.array(label)[idx])    

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

############
# training #
############

# evaluation metrics

def acc_at_k(y_true, y_pred, k=2):
    y_true = torch.tensor(y_true) if type(y_true) != torch.Tensor else y_true
    y_pred = torch.tensor(y_pred) if type(y_pred) != torch.Tensor else y_pred
    total = len(y_true)
    y_weights, y_idx = torch.topk(y_true, k=k, dim=-1)
    out_weights, out_idx = torch.topk(y_pred, k=k, dim=-1)
    correct = torch.sum(torch.eq(y_idx, out_idx) * y_weights)
    acc = correct / total
    return acc.item()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if len(labels.shape) > 1: 
        acc = acc_at_k(labels, predictions, k=2)
        return { 'accuracy': acc }        
    else:
        acc = accuracy_score(labels, predictions.argmax(-1))
        return { 'accuracy': acc } 

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
        
# trainers

class SibylTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        if len(labels.shape) > 1: 
            loss = CEwST_loss(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if return_outputs:
            return loss, outputs
        return loss


# callbacks

class TargetedMixturesCallback(TrainerCallback):
    """
    A callback that calculates a confusion matrix on the validation
    data and returns the most confused class pairings.
    """
    def __init__(self, dataloader, device, sentence1_key='text', sentence2_key=None, num_classes=2):
        self.dataloader = dataloader
        self.device = device
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.num_classes = num_classes
        
    def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
        cnf_mat = self.get_confusion_matrix(model, tokenizer, self.dataloader)
        new_targets = self.get_most_confused_per_class(cnf_mat)
        print("New targets:", new_targets)
        control = TrainerControl
        control.new_targets = new_targets
        if state.global_step < state.max_steps:
            control.should_training_stop = False
        else:
            control.should_training_stop = True
        return control
        
    def get_confusion_matrix(self, model, tokenizer, dataloader, normalize=True):
        n_classes = self.num_classes
        confusion_matrix = torch.zeros(n_classes, n_classes)
        with torch.no_grad():
            for batch in iter(self.dataloader):
                if 'input_ids' in batch: 
                    input_ids =  batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['labels']
                else:
                    if self.sentence2_key is None:
                        text1 = batch[sentence1_key]
                        text2 = None
                    else:
                        text1 = batch[sentence1_key]
                        text2 = batch[sentence2_key]
                    targets = batch['label']
                    data = tokenizer(text1, text2, padding=True, truncation=True, max_length=250, return_tensors='pt')
                    input_ids = data['input_ids'].to(self.device)
                    attention_mask = data['attention_mask'].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(outputs, dim=1).cpu()
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1    
            if normalize:
                confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=0)
        return confusion_matrix

    def get_most_confused_per_class(self, confusion_matrix):
        idx = torch.arange(len(confusion_matrix))
        cnf = confusion_matrix.fill_diagonal_(0).max(dim=1)[1]
        return torch.stack((idx, cnf)).T.tolist()