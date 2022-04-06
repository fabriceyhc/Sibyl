#!/usr/bin/env python
# coding: utf-8

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
import time
import random
import shutil
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sibyl import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir = "../../data1/fabricehc/prepared_datasets/"
save_dir = "../../data1/fabricehc/results_a100/"
local_save_dir = "./results/"

def tokenize_fn(batch):
    if sentence2_key is None:
        return tokenizer(batch[sentence1_key], padding=True, truncation=True, max_length=250)
    return tokenizer(batch[sentence1_key], batch[sentence2_key], padding=True, truncation=True, max_length=250)

def sibyl_tokenize_fn(text1, text2):
    if text2 is None:
        return tokenizer(text1, padding=True, truncation=True, max_length=250, return_tensors='pt')
    return tokenizer(text1, text2, padding=True, truncation=True, max_length=250, return_tensors='pt')


early_stopping_patience = 5

num_valid_per_class = 2000

num_runs = 3
num_train_examples = [10]
MODEL_NAMES = ['bert-base-uncased'] # ['bert-base-uncased', 'roberta-base', 'xlnet-base-cased']
ts = ['ORIG'] + [t.__name__ for t in TRANSFORMATIONS]
tasks = ['ag_news', 'imdb', 'dbpedia_14', 'yahoo_answers_topics', 'yelp_polarity', 'amazon_polarity']

run_args = []
for run_num in range(num_runs):
    for num_train_per_class in num_train_examples:
        for task in tasks:
            for MODEL_NAME in MODEL_NAMES:
                for t in ts:   
                    run_args.append({
                        "run_num":run_num,
                        "num_train_per_class":num_train_per_class,
                        "task":task,
                        "MODEL_NAME":MODEL_NAME,
                        "t":t
                    })
                 
results = []
save_file = 'transform_eval_all_10.csv'
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
            
    #############################################################
    ## Model + Tokenizer ########################################
    #############################################################
    checkpoint = local_save_dir + MODEL_NAME + '-' + task + '-' + t
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes).to(device)

    #############################################################
    ## Dataset Preparation ######################################
    #############################################################

    train_data_path = os.path.join(data_dir, task, 'ORIG', task + '_train_' + str(num_train_per_class))
    valid_data_path = os.path.join(data_dir, task, 'ORIG', task + '_valid_' + str(num_valid_per_class))

    train_dataset = load_from_disk(train_data_path)
    eval_dataset  = load_from_disk(valid_data_path)
    test_dataset  = load_dataset(task, split='test')

    # train_dataset = train_dataset.rename_column("label", "labels")

    # special handling for yahoo and dbpedia_14 datasets
    if task in ["dbpedia_14", "amazon_polarity"]:
        test_dataset = test_dataset.rename_column("content", "text")
    if task == "yahoo_answers_topics":
        test_dataset = test_dataset.map(lambda example : {'text' : example['question_title'] + " " + 
                                                                   example['question_content'] + " " +
                                                                   example['best_answer'],
                                                          'label': example['topic']})

    print('Length of train_dataset', num_classes * num_train_per_class, len(train_dataset))
    print('Length of valid_dataset', num_classes * num_valid_per_class, len(eval_dataset))

    assert num_classes * num_train_per_class == len(train_dataset)
    assert num_classes * num_valid_per_class == len(eval_dataset)

    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, batch_size=len(eval_dataset))
    eval_dataset = eval_dataset.rename_column("label", "labels") 
    eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_dataset = test_dataset.map(tokenize_fn, batched=True, batch_size=len(test_dataset))
    test_dataset = test_dataset.rename_column("label", "labels") 
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    #############################################################
    ## Callbacks + Collator #####################################
    #############################################################

    callbacks = []

    tmcb = None
    if early_stopping_patience > 0:
        escb = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience
        )
        callbacks.append(escb)

    transform = None
    num_sampled_INV = 0
    num_sampled_SIB = 0
    label_type = "soft"

    if t == "ORIG":
        label_type = "hard"
    elif t == "INV":
        num_sampled_INV = 2
        label_type = "hard"
    elif t == "SIB":
        num_sampled_SIB = 2
        label_type = "soft"
    elif t == 'INVSIB':
        num_sampled_INV = 1
        num_sampled_SIB = 1
        label_type = None
        label_type = "soft"
    else:
        transform = getattr(sys.modules[__name__], t)
        if hasattr(transform, 'uses_dataset'):
            transform = transform(dataset=task)
        else:
            transform = transform()
        if t in ['TextMix', 'SentMix', 'WordMix', 'ConceptMix']:
            tmcb = TargetedMixturesCallback(
                dataloader=DataLoader(eval_dataset, batch_size=4),
                device=device,
                sentence1_key=sentence1_key, 
                sentence2_key=sentence2_key,
                num_classes=num_classes
            )
            callbacks.append(tmcb)
            label_type = "soft"

        if sentence2_key is not None:
            continue # skip Mixture mutations for 2-input tasks

    collator = SibylCollator( 
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        tokenize_fn=sibyl_tokenize_fn, 
        transform=transform, 
        num_sampled_INV=num_sampled_INV, 
        num_sampled_SIB=num_sampled_SIB,
        dataset=task,
        num_outputs = 3,
        keep_original = True,
        task_type=task_type, 
        tran_type=None, 
        label_type=None,
        one_hot=label_type != "hard",
        transform_prob=1.0,
        target_pairs=[],
        target_prob=0.,
        reduce_mixed=False,
        num_classes=num_classes,
        return_tensors='pt',
        return_text=False
    )   

    #############################################################
    ## Trainer Setup ############################################
    #############################################################

    train_batch_size = 16
    eval_batch_size = 64
    num_epoch = 30
    gradient_accumulation_steps = 1
    max_steps = int((len(train_dataset) * num_epoch / gradient_accumulation_steps) / train_batch_size)
    logging_steps = max_steps // (num_epoch / 2)

    training_args = TrainingArguments(
        output_dir=checkpoint,
        overwrite_output_dir=True,
        max_steps=max_steps,
        save_steps=logging_steps,
        save_total_limit=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, 
        warmup_steps=int(max_steps / 10),
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        evaluation_strategy="steps",
        remove_unused_columns=False
    )

    trainer = SibylTrainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,                  
        train_dataset=train_dataset,         
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks
    )

    start_time = time.time()
    trainer.train()
    run_time = time.time() - start_time

    # test with ORIG data
    trainer.eval_dataset = test_dataset
    trainer.data_collator = DefaultCollator()
    if tmcb:    
        trainer.remove_callback(tmcb)

    out = trainer.evaluate()

    out['run_num'] = run_num
    out['num_train_per_class'] = num_train_per_class
    out['task'] = task
    out['transform'] = t
    out['run'] = checkpoint
    out['model_name'] = MODEL_NAME
    out['transform'] = t
    out['test'] = "ORIG"
    out['run_time'] = start_time
    print('ORIG for {}\n{}'.format(checkpoint, out))

    results.append(out)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(save_file)

    shutil.rmtree(checkpoint)