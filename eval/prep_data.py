from datasets import load_dataset, load_metric, concatenate_datasets, load_from_disk, Dataset

import os
import sys
import time
import random
import torch
import pandas as pd

import argparse

from sibyl import *

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

parser = argparse.ArgumentParser(description='Sibyl Dataset Generator')

parser.add_argument('--data-dir', type=str, default="/data1/fabricehc/prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--num-outputs', default=30, type=int, metavar='N',
                    help='augmentation multiplier - number of new inputs per 1 original')
parser.add_argument('--batch-size', default=10, type=int, metavar='N',
                    help='number of inputs to proccess per iteration')
parser.add_argument('--transforms', nargs='+', 
                    default=['ORIG', 'INV', 'SIB', 'INVSIB', 'Concept2Sentence', 
                             'TextMix', 'SentMix', 'WordMix'], 
                    type=str, help='transforms to apply from sibyl tool')
parser.add_argument('--tasks', nargs='+', 
                    default=['ag_news', 'amazon_polarity', 'yahoo_answers_topics', 
                             'imdb', 'yelp_polarity', 'dbpedia_14'], 
                    type=str, help='tasks (datasets) to train upon, from huggingface')
parser.add_argument('--num-train-per-class', nargs='+', default=[10, 200, 2500], type=int, 
                    help='number of training examples per class')
parser.add_argument('--num-valid-per-class', default=2000, type=int, metavar='N',
                    help='number of validation examples per class')
parser.add_argument('--save-file', type=str, default='prepare_dataset_log.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()


data_dir = args.data_dir
num_train_examples_per_class = args.num_train_per_class
num_valid_per_class = args.num_valid_per_class
num_outputs = args.num_outputs
tasks = args.tasks
ts = args.transforms

results = []
for task in tasks:
    
    # set task metadata
    task_to_keys = {
        "ag_news": {"keys": ("text", None), "num_classes": 4, "task_type": "topic"},
        "dbpedia_14": {"keys": ("text", None), "num_classes": 14, "task_type": "topic"},
        "yahoo_answers_topics": {"keys": ("text", None), "num_classes": 10, "task_type": "topic"},
        "imdb": {"keys": ("text", None), "num_classes": 2, "task_type": "sentiment"},
        "yelp_polarity":  {"keys": ("text", None), "num_classes": 2, "task_type": "sentiment"},
        "amazon_polarity":  {"keys": ("text", None), "num_classes": 2, "task_type": "sentiment"}
    }
    sentence1_key, sentence2_key = task_to_keys[task]["keys"]
    num_classes = task_to_keys[task]["num_classes"]
    task_type = task_to_keys[task]["task_type"]

    # load task dataset + preprocess
    dataset = load_dataset(task, split='train') 
    if task == "yahoo_answers_topics":
        dataset = dataset.map(lambda example : {'text' : example['question_title'] + " " + 
                                                         example['question_content'] + " " +
                                                         example['best_answer'],
                                                'label': example['topic']})

    if task in ["dbpedia_14", "amazon_polarity"]:
        dataset = dataset.rename_column("content", "text")

    dataset = dataset.remove_columns([c for c in list(dataset.features.keys()) if c not in ['text', 'label']])

    for num_train_per_class in num_train_examples_per_class:
        
        train_save_path = os.path.join(data_dir, task, 'ORIG', task + '_train_' + str(num_train_per_class))
        valid_save_path = os.path.join(data_dir, task, 'ORIG', task + '_valid_' + str(num_valid_per_class))
        
        if not os.path.exists(train_save_path):
            print("generating new, class-balanced datasets...")
            train_datasets = []
            valid_datasets = []
            for c in range(num_classes):
                class_dataset = dataset.filter(lambda example: example['label'] == c).shuffle()
                train_datasets.append(class_dataset.select(range(num_train_per_class)))
                if not os.path.exists(valid_save_path):
                    valid_datasets.append(class_dataset.select(range(num_train_per_class, num_valid_per_class + num_train_per_class)))

            train_dataset = concatenate_datasets(train_datasets).shuffle()
            train_dataset.save_to_disk(train_save_path)
            
            if not os.path.exists(valid_save_path):
                valid_dataset = concatenate_datasets(valid_datasets).shuffle()
                valid_dataset.save_to_disk(valid_save_path)  
                
        else:
            print("loading {}...".format(train_save_path))
            # abridged, class-balanced dataset already exists, so just load it
            train_dataset = load_from_disk(train_save_path)
        
        for t in ts:
            
            tran_train_save_path = os.path.join(data_dir, task, t, task + '_train_' + str(num_train_per_class))

            try:
                print(tran_train_save_path)
                pretransformed_dataset = load_from_disk(tran_train_save_path)
                print(pretransformed_dataset)
                print('example:', pretransformed_dataset[-1])
            except:
                pass

            multiplier = num_outputs if "Ada-" not in t and t != "ORIG" else 1
            keep_original = num_classes * num_train_per_class if "Ada-" not in t and t != "ORIG" else 0

            expected_len_train_dataset = (num_classes * num_train_per_class * multiplier) + keep_original
            
            if not os.path.exists(tran_train_save_path) or expected_len_train_dataset != len(pretransformed_dataset):

                transform = None
                num_sampled_INV = 0
                num_sampled_SIB = 0
                label_type = "hard"

                if t == "INV":
                    num_sampled_INV = 2
                elif t == "SIB":
                    num_sampled_SIB = 2
                    label_type = "soft"
                elif t == 'INVSIB':
                    num_sampled_INV = 1
                    num_sampled_SIB = 1
                    label_type = None
                else:
                    transform = getattr(sys.modules[__name__], t)
                    if hasattr(transform, 'uses_dataset'):
                        transform = transform(dataset=task)
                    else:
                        transform = transform()

                if t in ['TextMix', 'SentMix', 'WordMix', 'ConceptMix']:
                    label_type = "soft"

                sibyl_collator = SibylCollator( 
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    tokenize_fn=None, 
                    transform=transform, 
                    num_outputs=num_outputs,
                    keep_original=True,
                    num_sampled_INV=num_sampled_INV, 
                    num_sampled_SIB=num_sampled_SIB,
                    dataset=task,
                    task_type=task_type, 
                    tran_type=None, 
                    label_type=None,
                    one_hot=label_type != "hard",
                    transform_prob=1.0,
                    target_pairs=[],
                    target_prob=0.0,
                    reduce_mixed=False,
                    num_classes=num_classes,
                    return_tensors='np',
                    return_text=False
                )   

                train_dataset = train_dataset.shuffle()
                updated_dataset = train_dataset.map(sibyl_dataset_transform, batched=True, batch_size=args.batch_size)
                updated_dataset.save_to_disk(tran_train_save_path)

                out = {
                   "task": task, 
                   "num_classes": num_classes,
                   "transform": t,
                   "num_train_per_class": num_train_per_class,
                   "expected_len_train_dataset": expected_len_train_dataset, 
                   "len_train_dataset": len(train_dataset),
                   "match_len_train": expected_len_train_dataset == len(train_dataset),
                }
                results.append(out)
                df = pd.DataFrame(results)
                df.to_csv(args.save_file)

                print(updated_dataset)
                print('example:', updated_dataset[-1])
                