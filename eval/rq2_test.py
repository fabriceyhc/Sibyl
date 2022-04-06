from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    EarlyStoppingCallback
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_callback import TrainerControl
from datasets import load_dataset, load_metric, load_from_disk
import os
import sys
import argparse
import time
import random
import shutil
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sibyl import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def tokenize_fn(batch):
    if sentence2_key is None:
        return tokenizer(batch[sentence1_key], padding=True, truncation=True, max_length=250)
    return tokenizer(batch[sentence1_key], batch[sentence2_key], padding=True, truncation=True, max_length=250)


parser = argparse.ArgumentParser(description='Sibyl Tester')

parser.add_argument('--data-dir', type=str, default="../../data1/fabricehc/prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--save-dir', type=str, default="./results/testing/",
                    help='path to data folders')
parser.add_argument('--eval-batch-size', default=10, type=int, metavar='N',
                    help='eval batchsize')
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--transformers_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for TRANSFORMERS_CACHE')
parser.add_argument('--datasets_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for HF_DATASETS_CACHE')
parser.add_argument('--num_runs', default=100, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--num_tests', default=100, type=int, metavar='N',
                    help='number of tests to generate per run')
parser.add_argument('--transforms', nargs='+', 
                    default=['ORIG', 'INV', 'SIB', 'INVSIB', 'Concept2Sentence', 
                             'TextMix','SentMix','WordMix'], 
                    type=str, help='transforms to apply from sibyl tool')
parser.add_argument('--tasks', nargs='+', 
                    default=['ag_news', 'amazon_polarity', 'yahoo_answers_topics', 
                             'imdb', 'yelp_polarity', 'dbpedia_14'], 
                    type=str, help='tasks (datasets) to train upon, from huggingface')
parser.add_argument('--num-train-per-class', nargs='+', default=[2500], type=int, 
                    help='number of training examples per class')
parser.add_argument('--num-valid-per-class', default=2000, type=int, metavar='N',
                    help='number of validation examples per class')
parser.add_argument('--models', nargs='+',  
                    default=['fabriceyhc/bert-base-uncased-amazon_polarity',
                             'fabriceyhc/bert-base-uncased-yelp_polarity',
                             'fabriceyhc/bert-base-uncased-ag_news',
                             'fabriceyhc/bert-base-uncased-dbpedia_14',
                             'fabriceyhc/bert-base-uncased-yahoo_answers_topics',
                             'fabriceyhc/bert-base-uncased-imdb'], 
                    type=str, help='pretrained huggingface models to train')
parser.add_argument('--save-file', type=str, default='NLP_testing.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TRANSFORMERS_CACHE'] = args.transformers_cache
os.environ['HF_DATASETS_CACHE'] = args.datasets_cache

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir = args.data_dir
save_dir = args.save_dir

num_runs = args.num_runs
num_train_examples = args.num_train_per_class
test_models = args.models
ts = args.transforms
tasks = args.tasks
num_tests = args.num_tests

run_args = []
for num_train_per_class in num_train_examples:
    for t in ts:   
        for task in tasks:
            for MODEL_NAME in test_models:
                if task in MODEL_NAME:
                    run_args.append({
                        "num_train_per_class":num_train_per_class,
                        "task":task,
                        "MODEL_NAME":MODEL_NAME,
                        "t":t
                    })

results = []
save_file = args.save_file  
start_position = 0
if os.path.exists(save_file):
    results.extend(pd.read_csv(save_file).to_dict("records"))
    start_position = int(len(results) / num_tests)

print('starting at position {}'.format(start_position))
for run_arg in tqdm(run_args[start_position:]):
    num_train_per_class = run_arg['num_train_per_class']
    task = run_arg['task']
    MODEL_NAME = run_arg['MODEL_NAME']
    t = run_arg['t']

    print(pd.DataFrame([run_arg]))

    task_to_keys = {
        "ag_news": {"keys": ("text", None), "num_classes": 4, "task_type": "topic"},
        "dbpedia_14": {"keys": ("text", None), "num_classes": 14, "task_type": "topic"},
        "yahoo_answers_topics": {"keys": ("text", None), "num_classes": 10, "task_type": "topic"},
        "imdb": {"keys": ("text", None), "num_classes": 2, "task_type": "sentiment"},
        "amazon_polarity": {"keys": ("text", None), "num_classes": 2, "task_type": "sentiment"},
        "yelp_polarity": {"keys": ("text", None), "num_classes": 2, "task_type": "sentiment"}
    }
    sentence1_key, sentence2_key = task_to_keys[task]["keys"]
    num_classes = task_to_keys[task]["num_classes"]
    task_type = task_to_keys[task]["task_type"]
            
    #############################################################
    ## Model + Tokenizer ########################################
    #############################################################

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes).to(device)

    #############################################################
    ## Dataset Preparation ######################################
    #############################################################

    dataset_path = os.path.join(data_dir, task, t, task + '_train_' + str(num_train_per_class))
    if not os.path.exists(dataset_path):
        num_train_per_class = 200
        dataset_path = os.path.join(data_dir, task, t, task + '_train_' + str(num_train_per_class))
    dataset  = load_from_disk(dataset_path)

    dataset = dataset.map(tokenize_fn, batched=True, batch_size=len(dataset))
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    #############################################################
    ## Trainer Setup ############################################
    #############################################################

    testing_args = TrainingArguments(
        output_dir='./results/testing/',
        per_device_eval_batch_size=args.eval_batch_size ,
        remove_unused_columns=False,
        evaluation_strategy="no",
        disable_tqdm=True
    )

    for test_suite_id in range(num_runs):
        dataset_dict = dataset.train_test_split(test_size=num_tests, shuffle=True, seed=test_suite_id)
        test_suite = dataset_dict['test']
        trainer = SibylTrainer(
            model=model, 
            args=testing_args,
            eval_dataset=test_suite,
            compute_metrics=compute_metrics
        )
        out = trainer.evaluate()

        shutil.rmtree(os.path.join(save_dir, 'runs/'))

        # test with transformed data
        out['test_suite_id'] = test_suite_id
        out['num_train_per_class'] = num_train_per_class
        out['task'] = task
        out['transform'] = t
        out['model_name'] = MODEL_NAME
        out['transform'] = t

        results.append(out)
    
    # save results
    df = pd.DataFrame(results)
    df.to_csv(save_file)
