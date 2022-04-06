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

import textattack
from textattack import Attacker
from textattack.models.wrappers import ModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import (
    TextFoolerJin2019, 
    DeepWordBugGao2018, 
    Pruthi2019, 
    TextBuggerLi2018, 
    PSOZang2020, 
    CheckList2020, 
    BERTAttackLi2020
)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# https://www.gitmemory.com/issue/QData/TextAttack/424/795806095

def tokenize_fn(batch):
    if sentence2_key is None:
        return tokenizer(batch[sentence1_key], padding=True, truncation=True, max_length=250)
    return tokenizer(batch[sentence1_key], batch[sentence2_key], padding=True, truncation=True, max_length=250)

def sibyl_tokenize_fn(text1, text2):
    if text2 is None:
        return tokenizer(text1, padding=True, truncation=True, max_length=250, return_tensors='pt')
    return tokenizer(text1, text2, padding=True, truncation=True, max_length=250, return_tensors='pt')

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        out = []
        i = 0
        while i < len(text_input_list):
            batch = text_input_list[i : i + self.batch_size]
            encoding = self.tokenizer(batch, padding=True, truncation=True, max_length=250, return_tensors='pt')
            outputs = self.model(encoding['input_ids'].to(self.device), attention_mask=encoding['attention_mask'].to(self.device))
            preds = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu()
            out.append(preds)
            i += self.batch_size
        out = torch.cat(out)
        return out

parser = argparse.ArgumentParser(description='Sibyl Adversarial Attacker')

parser.add_argument('--data-dir', type=str, default="../../data1/fabricehc/prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--save-dir', type=str, default="./results/",
                    help='path to data folders')
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--transformers_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for TRANSFORMERS_CACHE')
parser.add_argument('--datasets_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for HF_DATASETS_CACHE')
parser.add_argument('--num_runs', default=3, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--transforms', nargs='+', 
                    default=['ORIG', 'INV', 'SIB', 'INVSIB', 'Concept2Sentence', 
                             'TextMix', 'Ada-TextMix', 'SentMix', 'Ada-SentMix', 
                             'WordMix', 'Ada-WordMix', 'ConceptMix'], 
                    type=str, help='transforms to apply from sibyl tool')
parser.add_argument('--tasks', nargs='+', 
                    default=['ag_news', 'amazon_polarity', 'yahoo_answers_topics', 
                             'imdb', 'yelp_polarity', 'dbpedia_14'], 
                    type=str, help='tasks (datasets) to train upon, from huggingface')
parser.add_argument('--num-train-per-class', nargs='+', default=[10], type=int, 
                    help='number of training examples per class')
parser.add_argument('--models', nargs='+',  default=['bert-base-uncased'], 
                    type=str, help='pretrained huggingface models to attack')
parser.add_argument('--save-file', type=str, default='NLP_adv_robustness_v2.csv',
                    help='name for the csv file to save with results')
parser.add_argument('--num_advs', default=100, type=int, metavar='N',
                    help='number of adversarial examples to generate')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TRANSFORMERS_CACHE'] = args.transformers_cache
os.environ['HF_DATASETS_CACHE'] = args.datasets_cache

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir = args.data_dir
save_dir = args.save_dir

num_runs = args.num_runs
num_train_examples = args.num_train_per_class
MODEL_NAMES = args.models
ts = args.transforms
tasks = args.tasks
num_outputs = 30
num_advs = args.num_advs

recipes = [PSOZang2020, CheckList2020, BERTAttackLi2020, TextFoolerJin2019, DeepWordBugGao2018, TextBuggerLi2018, ]

run_args = []
for run_num in range(num_runs):
    for num_train_per_class in num_train_examples:
        for t in ts:   
            for task in tasks:
                for MODEL_NAME in MODEL_NAMES:
                    run_args.append({
                        "run_num":run_num,
                        "num_train_per_class":num_train_per_class,
                        "task":task,
                        "MODEL_NAME":MODEL_NAME,
                        "t":t
                    })

results = []
save_file = args.save_file  
if os.path.exists(save_file):
    results.extend(pd.read_csv(save_file).to_dict("records"))
    start_position = len(results)
else:
    start_position = 0

print('starting at position {}'.format(start_position))
for run_arg in run_args[start_position:]:
    run_num = run_arg['run_num']
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

    test_dataset  = load_dataset(task, split='test')

    # special handling for yahoo and dbpedia_14 datasets
    if task in ["dbpedia_14", "amazon_polarity"]:
        test_dataset = test_dataset.rename_column("content", "text")
    if task == "yahoo_answers_topics":
        test_dataset = test_dataset.map(lambda example : {'text' : example['question_title'] + " " + 
                                                                   example['question_content'] + " " +
                                                                   example['best_answer'],
                                                          'label': example['topic']})
            
    #############################################################
    ## Model + Tokenizer ########################################
    #############################################################
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    checkpoint = save_dir + MODEL_NAME + '-' + task + '-' + t + '-' + str(num_train_per_class)
    loaded_checkpoint = False
    if os.path.exists(checkpoint):
        recent_checkpoint = [name for name in os.listdir(checkpoint) if 'checkpoint' in name]
        if recent_checkpoint:
            checkpoint = os.path.join(checkpoint, recent_checkpoint[-1])
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_classes).to(device)
        loaded_checkpoint = True

    #############################################################
    ## TextAttacks ##############################################
    #############################################################

    out = {}

    out['run_num'] = run_num
    out['num_train_per_class'] = num_train_per_class
    out['task'] = task
    out['transform'] = t
    out['run'] = checkpoint
    out['model_name'] = MODEL_NAME
    out['transform'] = t

    if loaded_checkpoint:

        mw = CustomModelWrapper(model, tokenizer)
        dataset = HuggingFaceDataset(test_dataset, shuffle=True)
        attack_args = textattack.AttackArgs(num_examples=num_advs, disable_stdout=True)

        for recipe in recipes:

            attack = recipe.build(mw)
            attacker = Attacker(attack, dataset, attack_args)
            attack_results = attacker.attack_dataset()

            num_results = 0
            num_failures = 0
            num_successes = 0

            for result in attack_results:
                
                # print(result.__str__(color_method='ansi'))
                
                num_results += 1
                if (type(result) == textattack.attack_results.SuccessfulAttackResult or 
                    type(result) == textattack.attack_results.MaximizedAttackResult):
                    num_successes += 1
                if type(result) == textattack.attack_results.FailedAttackResult:
                    num_failures += 1

            attack_success = num_successes / num_results
            out['attack_success_' + recipe.__name__] = attack_success

            print("{0} Attack Success: {1:0.2f}".format(recipe.__name__, attack_success))

    results.append(out)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(save_file)