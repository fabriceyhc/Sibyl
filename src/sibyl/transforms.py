from .transformations.text.contraction.expand_contractions import ExpandContractions
from .transformations.text.contraction.contract_contractions import ContractContractions
from .transformations.text.emoji.emojify import Emojify, AddPositiveEmoji, AddNegativeEmoji, AddNeutralEmoji
from .transformations.text.emoji.demojify import Demojify, RemovePositiveEmoji, RemoveNegativeEmoji, RemoveNeutralEmoji
from .transformations.text.negation.remove_negation import RemoveNegation
from .transformations.text.negation.add_negation import AddNegation
from .transformations.text.word_swap.change_number import ChangeNumber
from .transformations.text.word_swap.change_synse import ChangeSynonym, ChangeAntonym, ChangeHyponym, ChangeHypernym
from .transformations.text.word_swap.word_deletion import WordDeletion
from .transformations.text.word_swap.homoglyph_swap import HomoglyphSwap
from .transformations.text.word_swap.random_swap import RandomSwap
from .transformations.text.insertion.random_insertion import RandomInsertion
from .transformations.text.insertion.sentiment_phrase import InsertSentimentPhrase, InsertPositivePhrase, InsertNegativePhrase
from .transformations.text.insertion.insert_punctuation_marks import InsertPunctuationMarks
from .transformations.text.links.add_sentiment_link import AddSentimentLink, AddPositiveLink, AddNegativeLink
from .transformations.text.links.import_link_text import ImportLinkText
from .transformations.text.entities.change_location import ChangeLocation
from .transformations.text.entities.change_name import ChangeName
from .transformations.text.typos.char_delete import RandomCharDel
from .transformations.text.typos.char_insert import RandomCharInsert
from .transformations.text.typos.char_substitute import RandomCharSubst
from .transformations.text.typos.char_swap import RandomCharSwap
from .transformations.text.typos.char_swap_qwerty import RandomSwapQwerty 
from .transformations.text.mixture.text_mix import TextMix, SentMix, WordMix
from .transformations.text.generative.concept2sentence import Concept2Sentence
from .transformations.text.mixture.concept_mix import ConceptMix

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
    InsertPunctuationMarks,
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
    WordMix,
    Concept2Sentence,
    ConceptMix
]

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

from .utils import *
from .transformations.utils import *

def init_transforms(task_type=None, tran_type=None, label_type=None, return_metadata=True, dataset=None, transforms=None):
    df_all = []
    if transforms:
        for tran in transforms:
            df = tran.get_task_configs()
            df['transformation'] = tran.__class__.__name__
            df['tran_fn'] = tran
            df_all.append(df)
    else:
        for tran in TRANSFORMATIONS:
            if hasattr(tran, 'uses_dataset'):
                t = tran(return_metadata=return_metadata, dataset=dataset)
            else:
                t = tran(return_metadata=return_metadata)
            df = t.get_task_configs()
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
        num_outputs       : the number of transformed sentences per input sentence
        keep_original     : wheter to keep the original (untransformed) sentence in the augmented set; if num_outputs == 1, then false
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
                 num_outputs=1,
                 keep_original=True,
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
        self.num_outputs = num_outputs
        self.keep_original = keep_original
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
        try:
            if torch.rand(1) < self.transform_prob:
                if self.transform:
                    if 'AbstractBatchTransformation' in self.transform.__class__.__bases__[0].__name__:
                        new_text, new_labels = [], []
                        batch = (text, labels)
                        for _ in range(self.num_outputs):
                            new_batch = self.transform(
                                batch, 
                                self.target_pairs,   
                                self.target_prob,
                                self.num_classes
                            )
                            new_text.extend(new_batch[0])
                            new_labels.extend(new_batch[1])
                    else:
                        task_config = init_transforms(task_type=self.task_type, 
                                                      tran_type=self.tran_type, 
                                                      label_type=self.label_type, 
                                                      return_metadata=True,
                                                      dataset=self.dataset,
                                                      transforms=[self.transform]).to_dict(orient='records')[0]
                        new_text, new_labels = [], []
                        for _ in range(self.num_outputs):
                            for X, y in zip(text, labels):
                                X, y = self.transform.transform_Xy(X, y, task_config)
                                new_text.append(X)
                                new_labels.append(y)               
                else:
                    new_text, new_labels, trans = [], [], []
                    for _ in range(self.num_outputs):
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

                if self.keep_original:
                    text.extend(new_text)
                    if self.one_hot:
                        labels = [one_hot_encode(y, self.num_classes) for y in labels]
                    labels.extend(new_labels)
                else:                  
                    text = new_text   
                    labels = new_labels
                    
        except Exception:
            traceback.print_exc()

        if self.sentence2_key is None:
            text1 = [str(x[0]) if type(x) == list else str(x) for x in text]
            text2 = None
        else:
            text1 = [str(x[0]) if type(x) == list else str(x) for x in text]
            text2 = [str(x[1]) if type(x) == list else str(x) for x in text]
        labels = [np.squeeze(y).tolist() if isinstance(y, (list, np.ndarray, torch.Tensor)) else y for y in labels]
        
        if self.reduce_mixed and len(labels.shape) >= 2:
            labels = [np.argmax(y) if i == 1 else self.num_classes for (i, y) in zip(np.count_nonzero(labels, axis=-1), labels)]

        if self.one_hot and len(np.array(labels).shape) == 1 and not self.reduce_mixed:
            # print([type(l) for l in labels])
            labels = [one_hot_encode(y, self.num_classes) if isinstance(y, (int, np.int64)) else y for y in labels]
            # print(labels)
            labels = torch.tensor(labels, dtype=torch.int64)
            # labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)

        if self.return_tensors == 'np':
            labels = np.array(labels)
        if self.return_tensors == 'pt':
            # print(labels)
            labels = torch.tensor(labels)
            if len(labels.shape) == 1:
                labels = labels.long()
                
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

def sibyl_dataset_transform(batch):
    """
    To be used with datasets.Dataset.map()
    Example:
    `
    updated_dataset = dataset.map(sibyl_dataset_transform, 
                                  batched=True, 
                                  batch_size=batch_size)
    `
    """
    new_batch = []
    for data, target in zip(batch['text'], batch['label']):
        new_batch.append({'text': data, 'label': target})
    text, label = sibyl_collator(new_batch)
    return {"text": text, "label": label}