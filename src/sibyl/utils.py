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
    if task == "stsb":
        predictions = predictions[:, 0]
        return pearson.compute(predictions=predictions, references=labels)
    # elif task == "cola":
    #     predictions = np.argmax(predictions, axis=1)
    #     return matthews.compute(predictions=predictions, references=labels)
    else:
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
        

# collators

class DefaultCollator:
    def __init__(self):
        pass
    def __call__(self, batch):
        return torch.utils.data.dataloader.default_collate(batch)

class SibylCollator:
    """
        sentence1_key     : the key to the first input (required)
        sentence2_key     : the key to the second input (optional)
        tokenize_fn       : tokenizer, if none provided, just returns inputs and targets
        transform         : a particular initialized transform (e.g. TextMix()) to apply to the inputs (overrides random sampling)
        num_sampled_INV   : number of uniformly sampled INV transforms to apply upon the inputs 
        num_sampled_SIB   : number of uniformly sampled SIB transforms to apply upon the inputs 
        task_type         : filter on transformations for task type [topic, sentiment]
        tran_type         : filter on transformations for tran type [INV, SIB]
        label_type        : filter on transformations for task type [hard, soft]
        one_hot           : whether or not to one-hot-encode the targets
        transform_prob    : probability of transforming the inputs at all
        target_pairs      : the class sets to mix
        target_prob       : probability of targeting mixture mutations for particular classes
        reduce_mixed      : label previously mixed classes to be a new-hard label class
        num_classes       : number of classes, used for one-hot-encoding
        return_tensors    : return type of the labels, only supports numpy (np) and pytorch (pt), 
                            the return type of other batch content is controlled by tokenize_fn
        return_text       : the tokenize_fn typically replaces the raw text with the tokenized representation,
                            setting this value to true adds back the raw text for additional use
    """
    def __init__(self, 
                 sentence1_key,
                 sentence2_key=None,
                 tokenize_fn=None, 
                 transform=None, 
                 num_sampled_INV=0, 
                 num_sampled_SIB=0, 
                 dataset=None,
                 task_type=None, 
                 tran_type=None, 
                 label_type=None,
                 one_hot=False,
                 transform_prob=1.0, 
                 target_pairs=[], 
                 target_prob=1.0, 
                 reduce_mixed=False,
                 num_classes=2,
                 return_tensors='pt',
                 return_text=False):
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.tokenize_fn = tokenize_fn
        self.transform = transform
        self.num_sampled_INV = num_sampled_INV
        self.num_sampled_SIB = num_sampled_SIB
        self.dataset = dataset
        self.task_type = task_type
        self.tran_type = tran_type
        self.label_type = label_type
        self.one_hot = one_hot
        self.transform_prob = transform_prob
        self.target_pairs = target_pairs
        self.target_prob = target_prob
        self.reduce_mixed = reduce_mixed
        self.num_classes = num_classes
        self.return_tensors = return_tensors
        self.return_text = return_text
        
        if self.transform: 
                print("SibylCollator initialized with {}".format(transform.__class__.__name__))
                
        else: 
            self.transforms_df = init_transforms(task_type=task_type, 
                                                 tran_type=tran_type, 
                                                 label_type=label_type, 
                                                 return_metadata=True,
                                                 dataset=dataset)
            print("SibylCollator initialized with num_sampled_INV={} and num_sampled_SIB={}".format(
                num_sampled_INV, num_sampled_SIB))
        
        
    def __call__(self, batch):
        if all(['input_ids' in x for x in batch]):
            return {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'labels': torch.stack([x['labels'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch])
            }
        if self.sentence2_key is None:
            text = [x[self.sentence1_key] for x in batch]
        else:
            text = [[x[self.sentence1_key], x[self.sentence2_key]] for x in batch]
        labels = [x['label'] for x in batch]
        if torch.rand(1) < self.transform_prob:
            if self.transform:
                if 'AbstractBatchTransformation' in self.transform.__class__.__bases__[0].__name__:
                    text, labels = self.transform(
                        (text, labels), 
                        self.target_pairs,   
                        self.target_prob,
                        self.num_classes
                    )
                else:
                    task_config = init_transforms(task_type=self.task_type, 
                                                  tran_type=self.tran_type, 
                                                  label_type=self.label_type, 
                                                  return_metadata=True,
                                                  dataset=dataset,
                                                  transforms=[self.transform]).to_dict(orient='records')[0]
                    new_text, new_labels = [], []
                    for X, y in zip(text, labels):
                        X, y = self.transform.transform_Xy(X, y, task_config)
                        new_text.append(X)
                        new_labels.append(y)           
                    text = new_text   
                    labels = new_labels     
            else:
                new_text, new_labels, trans = [], [], []
                for X, y in zip(text, labels): 
                    t_trans = []
                    
                    num_tries = 0
                    num_INV_applied = 0
                    while num_INV_applied < self.num_sampled_INV:
                        if num_tries > 25:
                            break
                        t_df   = self.transforms_df[self.transforms_df['tran_type']=='INV'].sample(1)
                        t_fn   = t_df['tran_fn'].iloc[0]
                        t_name = t_df['transformation'].iloc[0]  
                        t_config = t_df.to_dict(orient='records')[0]
                        if t_name in trans:
                            continue
                        try:
                            X, y, meta = t_fn.transform_Xy(X, y, t_config)
                        except Exception as e:
                            meta = {"change":False}
                            print(e)
                        if self.one_hot:
                            y = one_hot_encode(y, self.num_classes)
                        if meta['change']:
                            num_INV_applied += 1
                            t_trans.append(t_name)
                        num_tries += 1

                    num_tries = 0
                    num_SIB_applied = 0       
                    while num_SIB_applied < self.num_sampled_SIB:
                        if num_tries > 25:
                            break
                        t_df   = self.transforms_df[self.transforms_df['tran_type']=='SIB'].sample(1)
                        t_fn   = t_df['tran_fn'].iloc[0]
                        t_name = t_df['transformation'].iloc[0]   
                        t_config = t_df.to_dict(orient='records')[0]  
                        if t_name in trans:
                            continue
                        if 'AbstractBatchTransformation' in t_fn.__class__.__bases__[0].__name__:
                            Xs, ys = sample_Xy(text, labels, num_sample=1)
                            Xs.append(X); ys.append(y) 
                            # Xs = [str(x) for x in Xs]
                            ys = [np.squeeze(one_hot_encode(y, self.num_classes)) for y in ys]
                            (X, y), meta = t_fn((Xs, ys), self.target_pairs, self.target_prob, self.num_classes)
                            X, y = X[0], y[0]
                        else:
                            try:
                                X, y, meta = t_fn.transform_Xy(X, y, t_config)
                            except Exception as e:
                                meta = {"change":False}
                                print(e)
                            if self.one_hot:
                                y = one_hot_encode(y, self.num_classes)
                        if meta['change']:
                            num_SIB_applied += 1
                            t_trans.append(t_name)
                        num_tries += 1

                    new_text.append(X)
                    new_labels.append(y)
                    trans.append(t_trans)            
                text = new_text   
                labels = new_labels

        if self.sentence2_key is None:
            text1 = [x[0] if type(x) == list else x for x in text]
            text2 = None
        else:
            text1 = [x[0] for x in text]
            text2 = [x[1] for x in text]
        labels = [y.squeeze().tolist() if type(y) == np.ndarray else y for y in labels]
        
        if self.reduce_mixed and len(labels.shape) >= 2:
            labels = [np.argmax(y) if i == 1 else self.num_classes for (i, y) in zip(np.count_nonzero(labels, axis=-1), labels)]

        if self.return_tensors == 'np':
            labels = np.array(labels)
        if self.return_tensors == 'pt':
            labels = torch.tensor(labels)
            if len(labels.shape) == 1:
                labels = labels.long()

        if self.one_hot and len(labels.shape) == 1 and not self.reduce_mixed:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)

        if self.tokenize_fn:
            batch = self.tokenize_fn(text1, text2)
            batch['labels'] = labels
            batch.pop('idx', None)
            batch.pop('label', None)
            if self.return_text:
                batch['text'] = text
        else:
            if self.sentence2_key is None:
                batch = (text1, labels)
            else:
                batch = (text1, text2, labels)
        return batch

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