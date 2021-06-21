from transformations.text.contraction.expand_contractions import ExpandContractions
from transformations.text.contraction.contract_contractions import ContractContractions
from transformations.text.emoji.emojify import Emojify, AddPositiveEmoji, AddNegativeEmoji, AddNeutralEmoji
from transformations.text.emoji.demojify import Demojify, RemovePositiveEmoji, RemoveNegativeEmoji, RemoveNeutralEmoji
from transformations.text.negation.remove_negation import RemoveNegation
from transformations.text.negation.add_negation import AddNegation
from transformations.text.contraction.expand_contractions import ExpandContractions
from transformations.text.contraction.contract_contractions import ContractContractions
from transformations.text.word_swap.change_number import ChangeNumber
from transformations.text.word_swap.change_synse import ChangeSynonym, ChangeAntonym, ChangeHyponym, ChangeHypernym
from transformations.text.word_swap.word_deletion import WordDeletion
from transformations.text.word_swap.homoglyph_swap import HomoglyphSwap
from transformations.text.word_swap.random_swap import RandomSwap
from transformations.text.insertion.random_insertion import RandomInsertion
from transformations.text.insertion.sentiment_phrase import InsertSentimentPhrase, InsertPositivePhrase, InsertNegativePhrase
from transformations.text.links.add_sentiment_link import AddSentimentLink, AddPositiveLink, AddNegativeLink
from transformations.text.links.import_link_text import ImportLinkText
from transformations.text.entities.change_location import ChangeLocation
from transformations.text.entities.change_name import ChangeName
from transformations.text.typos.char_delete import RandomCharDel
from transformations.text.typos.char_insert import RandomCharInsert
from transformations.text.typos.char_substitute import RandomCharSubst
from transformations.text.typos.char_swap import RandomCharSwap
from transformations.text.typos.char_swap_qwerty import RandomSwapQwerty 
from transformations.text.mixture.text_mix import TextMix, SentMix, WordMix

TRANSFORMATIONS = [
    ExpandContractions,
    ContractContractions,
    Emojify,
    AddPositiveEmoji,
    AddNegativeEmoji,
    AddNeutralEmoji,
    Demojify, 
    RemovePositiveEmoji,
    RemoveNegativeEmoji,
    RemoveNeutralEmoji,
    ChangeLocation,
    ChangeName,
    InsertPositivePhrase,
    InsertNegativePhrase,
    RandomInsertion,
    AddPositiveLink,
    AddNegativeLink,
    ImportLinkText,
    AddNegation,
    RemoveNegation,
    RandomCharDel,
    RandomCharInsert, 
    RandomCharSubst, 
    RandomCharSwap, 
    RandomSwapQwerty,
    ChangeNumber,
    ChangeSynonym, 
    ChangeAntonym, 
    ChangeHyponym, 
    ChangeHypernym,
    WordDeletion, 
    HomoglyphSwap, 
    RandomSwap, 
    TextMix, 
    SentMix, 
    WordMix
]

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *
from transformations.utils import *

def init_transforms(task_type=None, tran_type=None, label_type=None, meta=True):
    df_all = []
    for transform in TRANSFORMATIONS:
        t = transform(task=task_type, meta=meta)
        df = t.get_tran_types()
        df['transformation'] = t.__class__.__name__
        df['tran_fn'] = t
        df_all.append(df)
    df = pd.concat(df_all)
    if task_type is not None:
        task_df = df['task_name'] == task_type
        df = df[task_df]
    if tran_type is not None:
        tran_df = df['tran_type'] == tran_type
        df = df[tran_df]
    if label_type is not None:
        label_df = df['label_type'] == label_type
        df = df[label_df]
    df.reset_index(drop=True, inplace=True)
    return df

def transform_test_suites(
    test_suites, 
    num_INV_required=1, 
    num_SIB_required=1, 
    task_type=None, 
    tran_type=None, 
    label_type=None,
    one_hot=True):
    
    df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)

    new_test_suites = {}
    for i, test_suite in tqdm(test_suites.items()):
        new_text, new_label, trans = [], [], []
        text, label = test_suite.items()
        text, label = text[1], label[1]
        num_classes = len(np.unique(label))   
        for X, y in zip(text, label): 
            t_trans = []
            num_tries = 0
            num_INV_applied = 0
            while num_INV_applied < num_INV_required:
                if num_tries > 25:
                    break
                t_df   = df[df['tran_type']=='INV'].sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]                
                if t_name in trans:
                    continue
                X, y, meta = t_fn.transform_Xy(str(X), y)
                if one_hot:
                    y = one_hot_encode(y, num_classes)
                if meta['change']:
                    num_INV_applied += 1
                    t_trans.append(t_name)
                num_tries += 1

            num_tries = 0
            num_SIB_applied = 0       
            while num_SIB_applied < num_SIB_required:
                if num_tries > 25:
                    break
                t_df   = df[df['tran_type']=='SIB'].sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]                
                if t_name in trans:
                    continue
                if 'AbstractBatchTransformation' in t_fn.__class__.__bases__[0].__name__:
                    Xs, ys = sample_Xy(text, label, num_sample=1)
                    Xs.append(X); ys.append(y) 
                    Xs = [str(x) for x in Xs]
                    ys = [np.squeeze(one_hot_encode(y, num_classes)) for y in ys]
                    (X, y), meta = t_fn((Xs, ys), num_classes=num_classes)
                    X, y = X[0], y[0]
                else:
                    X, y, meta = t_fn.transform_Xy(str(X), y)
                if meta['change']:
                    num_SIB_applied += 1
                    t_trans.append(t_name)
                num_tries += 1

            new_text.append(X)
            new_label.append(y)
            trans.append(t_trans)
        
        new_test_suites[i] = {'data': new_text, 'target': new_label, 'ts': trans}
        
    return new_test_suites

def transform_dataset(dataset, num_transforms=2, task_type=None, tran_type=None, label_type=None):
    df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)
    text, label = dataset['text'], dataset['label'] 
    new_text, new_label, trans = [], [], []
    if tran_type == 'SIB':
        if type(text) == list:
            text = np.array(text, dtype=np.string_)
            label = pd.get_dummies(label).to_numpy(dtype=np.float)
        batch_size= 1000
        for i in tqdm(range(0, len(label), batch_size)):
            text_batch = text[i:i+batch_size]
            label_batch = label[i:i+batch_size]
            batch = (text_batch, label_batch)
            t_trans = []
            n = 0
            num_tries = 0
            while n < num_transforms:
                if num_tries > 25:
                    break
                t_df   = df.sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]
                if t_name in trans:
                    continue
                else:
                    t_trans.append(t_name)
                batch, meta = t_fn(batch)
                if meta['change']:
                    n += 1
                else:
                    t_trans.remove(t_name)
                num_tries += 1
                if len(batch[0].shape) > 1:
                    # print(t_name)
                    break
            new_text.extend(batch[0].tolist())
            new_label.extend(batch[1].tolist())
            trans.append(t_trans)
    else:
        for X, y in tqdm(zip(text, label), total=len(label)):
            t_trans = []
            n = 0
            num_tries = 0
            while n < num_transforms:
                if num_tries > 25:
                    break
                t_df   = df.sample(1)
                t_fn   = t_df['tran_fn'].iloc[0]
                t_name = t_df['transformation'].iloc[0]
                if t_name in t_trans:
                    continue
                else:
                    t_trans.append(t_name)
                X, y, meta = t_fn.transform_Xy(str(X), y)
                if meta['change']:
                    n += 1
                else:
                    t_trans.remove(t_name)
            new_text.append(X)
            new_label.append(y)
            trans.append(t_trans)
    return new_text, new_label, trans

def transform_dataset_INVSIB(
    dataset, 
    num_INV_required=1, 
    num_SIB_required=1, 
    task_type=None, 
    tran_type=None, 
    label_type=None,
    one_hot=True):
    
    df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)
    
    text, label = dataset['text'], dataset['label']
    new_text, new_label, trans = [], [], []

    num_classes = len(np.unique(label))
    
    for X, y in tqdm(zip(text, label), total=len(label)): 
        t_trans = []

        num_tries = 0
        num_INV_applied = 0
        while num_INV_applied < num_INV_required:
            if num_tries > 25:
                break
            t_df   = df[df['tran_type']=='INV'].sample(1)
            t_fn   = t_df['tran_fn'].iloc[0]
            t_name = t_df['transformation'].iloc[0]                
            if t_name in trans:
                continue
            X, y, meta = t_fn.transform_Xy(str(X), y)
            if one_hot:
                y = one_hot_encode(y, num_classes)
            if meta['change']:
                num_INV_applied += 1
                t_trans.append(t_name)
            num_tries += 1

        num_tries = 0
        num_SIB_applied = 0       
        while num_SIB_applied < num_SIB_required:
            if num_tries > 25:
                break
            t_df   = df[df['tran_type']=='SIB'].sample(1)
            t_fn   = t_df['tran_fn'].iloc[0]
            t_name = t_df['transformation'].iloc[0]                
            if t_name in trans:
                continue
            if 'AbstractBatchTransformation' in t_fn.__class__.__bases__[0].__name__:
                Xs, ys = sample_Xy(text, label, num_sample=1)
                Xs.append(X); ys.append(y) 
                Xs = [str(x) for x in Xs]
                ys = [np.squeeze(one_hot_encode(y, num_classes)) for y in ys]
                (X, y), meta = t_fn((Xs, ys),num_classes=num_classes)
                X, y = X[0], y[0]
            else:
                X, y, meta = t_fn.transform_Xy(str(X), y)
            if meta['change']:
                num_SIB_applied += 1
                t_trans.append(t_name)
            num_tries += 1

        new_text.append(X)
        new_label.append(y)
        trans.append(t_trans)
                
    new_text = [str(x).encode('utf-8') for x in new_text]
    return np.array(new_text, dtype=np.string_), np.array(new_label), np.array(trans, dtype=np.string_)


class SibylCollator:
    """
        tokenize_fn       : tokenizer 
        transform         : a particular initialized transform (e.g. TextMix()) to apply to the inputs (overrides random sampling)
        num_sampled_INV   : number of uniformly sampled INV transforms to apply upon the inputs 
        num_sampled_SIB=1 : number of uniformly sampled SIB transforms to apply upon the inputs 
        task_type         : filter on transformations for task type [topic, sentiment]
        tran_type         : filter on transformations for tran type [INV, SIB]
        label_type        : filter on transformations for task type [hard, soft]
        one_hot           : whether or not to one-hot-encode the targets
        transform_prob    : probability of transforming the inputs at all
        target_pairs      : the class sets to mix
        target_prob       : probability of targeting mixture mutations for particular classes
        reduce_mixed      : label previously mixed classes to be a new-hard label class
        num_classes       : number of classes, used for one-hot-encoding
    """
    def __init__(self, 
                 tokenize_fn, 
                 transform=None, 
                 num_sampled_INV=0, 
                 num_sampled_SIB=0, 
                 task_type=None, 
                 tran_type=None, 
                 label_type=None,
                 one_hot=True,
                 transform_prob=1.0, 
                 target_pairs=[], 
                 target_prob=1.0, 
                 reduce_mixed=False,
                 num_classes=2):
        
        self.tokenize_fn = tokenize_fn
        self.transform = transform
        self.num_sampled_INV = num_sampled_INV
        self.num_sampled_SIB = num_sampled_SIB
        self.task_type = task_type
        self.tran_type = tran_type
        self.label_type = label_type
        self.one_hot = one_hot
        self.transform_prob = transform_prob
        self.target_pairs = target_pairs
        self.target_prob = target_prob
        self.reduce_mixed = reduce_mixed
        self.num_classes = num_classes

        self.transforms_df = init_transforms(task_type=task_type, tran_type=tran_type, label_type=label_type, meta=True)

        if (not transform and num_sampled_INV == 0 and num_sampled_SIB == 0) or (len(self.transforms_df) == 0):
            raise ValueError("Must specify a particular transform or a value for num_sampled_INV / num_sampled_SIB, otherwise no transforms will be applied.")

        if self.transform:
            print("SibylCollator initialized with {}".format(transform.__class__.__name__))
        else: 
            print("SibylCollator initialized with num_sampled_INV={} and num_sampled_SIB={}".format(num_sampled_INV, num_sampled_SIB))

        
    def __call__(self, batch):
        text = [x['text'] for x in batch]
        labels = [x['label'] for x in batch]
        if torch.rand(1) < self.transform_prob:


            if self.transform:
                text, labels = self.transform(
                    (text, labels), 
                    self.target_pairs,   
                    self.target_prob,
                    self.num_classes
                )
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
                        if t_name in trans:
                            continue
                        X, y, meta = t_fn.transform_Xy(str(X), y)
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
                        if t_name in trans:
                            continue
                        if 'AbstractBatchTransformation' in t_fn.__class__.__bases__[0].__name__:
                            Xs, ys = sample_Xy(text, labels, num_sample=1)
                            Xs.append(X); ys.append(y) 
                            Xs = [str(x) for x in Xs]
                            ys = [np.squeeze(one_hot_encode(y, self.num_classes)) for y in ys]
                            (X, y), meta = t_fn((Xs, ys), self.target_pairs, self.target_prob, self.num_classes)
                            X, y = X[0], y[0]
                        else:
                            X, y, meta = t_fn.transform_Xy(str(X), y)
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

        labels = np.array(labels)  
        if self.reduce_mixed and len(labels.shape) >= 2:
            labels = torch.tensor([np.argmax(y) if i == 1 else self.num_classes for (i, y) in zip(np.count_nonzero(labels, axis=-1), labels)], dtype=torch.long)
        else:
            labels = torch.tensor(labels)

        if len(labels.shape) == 1:
            labels = labels.long()

        if self.one_hot and len(labels.shape) == 1 and not self.reduce_mixed:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
        batch = self.tokenize_fn(text)
        batch['labels'] = labels
        batch.pop('idx', None)
        batch.pop('label', None)
        return batch