# NLP | Train

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

parser = argparse.ArgumentParser(description='Sibyl Trainer')

parser.add_argument('--data-dir', type=str, default="/data1/fabricehc/prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--save-dir', type=str, default="/data1/fabricehc/pretrained/",
                    help='path to data folders')
parser.add_argument('--num_epoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--eval-batch-size', default=32, type=int, metavar='N',
                    help='eval batchsize')
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--transformers_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for TRANSFORMERS_CACHE')
parser.add_argument('--datasets_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for HF_DATASETS_CACHE')
parser.add_argument('--num_runs', default=1, type=int, metavar='N',
                    help='number of times to repeat the training')
parser.add_argument('--transforms', nargs='+', 
                    default=['ORIG', 'INV', 'SIB', 'INVSIB', 'Concept2Sentence', 
                             'TextMix', 'Ada-TextMix', 'SentMix', 'Ada-SentMix', 
                             'WordMix', 'Ada-WordMix', 'ConceptMix'], 
                    type=str, help='transforms to apply from sibyl tool')
parser.add_argument('--tasks', nargs='+', 
                    default=['amazon_polarity', 'yahoo_answers_topics', 
                             'dbpedia_14', 'imdb', 'yelp_polarity', 'ag_news'], 
                    type=str, help='tasks (datasets) to train upon, from huggingface')
parser.add_argument('--num-train-per-class', nargs='+', default=[10, 200, 2500], type=int, 
                    help='number of training examples per class')
parser.add_argument('--num-valid-per-class', default=2000, type=int, metavar='N',
                    help='number of validation examples per class')
parser.add_argument('--num-outputs', default=30, type=int, metavar='N',
                    help='augmentation multiplier - number of new inputs per 1 original')
parser.add_argument('--models', nargs='+',  default=['bert-base-uncased'], 
                    type=str, help='pretrained huggingface models to train')
parser.add_argument('--save-file', type=str, default='NLP_training_pregen.csv',
                    help='name for the csv file to save with results')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TRANSFORMERS_CACHE'] = args.transformers_cache
os.environ['HF_DATASETS_CACHE'] = args.datasets_cache

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def tokenize_fn(batch):
    if sentence2_key is None:
        return tokenizer(batch[sentence1_key], padding=True, truncation=True, max_length=250)
    return tokenizer(batch[sentence1_key], batch[sentence2_key], padding=True, truncation=True, max_length=250)

def sibyl_tokenize_fn(text1, text2):
    if text2 is None:
        return tokenizer(text1, padding=True, truncation=True, max_length=250, return_tensors='pt')
    return tokenizer(text1, text2, padding=True, truncation=True, max_length=250, return_tensors='pt')


data_dir = args.data_dir
save_dir = args.save_dir
num_runs = args.num_runs
num_train_examples = args.num_train_per_class
num_valid_per_class = args.num_valid_per_class
MODEL_NAMES = args.models
ts = args.transforms
tasks = args.tasks
num_outputs = args.num_outputs

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

print(run_args)
                 
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
            
    #############################################################
    ## Model + Tokenizer ########################################
    #############################################################
    checkpoint = save_dir + MODEL_NAME + '-' + task + '-' + t + '-' + str(num_train_per_class)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes).to(device)

    #############################################################
    ## Dataset Preparation ######################################
    #############################################################

    if "Ada-" in t:
        train_data_path = os.path.join(data_dir, task, 'ORIG', task + '_train_' + str(num_train_per_class))
    else:
        train_data_path = os.path.join(data_dir, task, t, task + '_train_' + str(num_train_per_class))
    valid_data_path = os.path.join(data_dir, task, 'ORIG', task + '_valid_' + str(num_valid_per_class))

    train_dataset = load_from_disk(train_data_path).shuffle()
    eval_dataset  = load_from_disk(valid_data_path)
    test_dataset  = load_dataset(task, split='test')

    # special handling for certain datasets
    if task in ["dbpedia_14", "amazon_polarity"]:
        test_dataset = test_dataset.rename_column("content", "text")
    if task == "yahoo_answers_topics":
        test_dataset = test_dataset.map(lambda example : {'text' : example['question_title'] + " " + 
                                                                   example['question_content'] + " " +
                                                                   example['best_answer'],
                                                          'label': example['topic']})

    print('Length of train_dataset', len(train_dataset))
    print('Length of eval_dataset', len(eval_dataset))

    multiplier = 1
    keep_original = 0
    if "Ada-" not in t and t != "ORIG":
        multiplier = num_outputs 
        keep_original = num_classes * num_train_per_class
    expected_len_train_dataset = (num_classes * num_train_per_class * multiplier) + keep_original
    assert expected_len_train_dataset == len(train_dataset), "Unexpected number of training examples"

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
    escb = EarlyStoppingCallback(
        early_stopping_patience=10
    )
    callbacks.append(escb)

    transform = None
    transform_prob = 0
    num_sampled_INV = 0
    num_sampled_SIB = 0
    label_type = "hard"
    keep_original = True

    if t not in ["ORIG", "INV", "EDA", "AEDA", "Concept2Sentence"]:
        label_type = "soft"

    if t in ['Ada-TextMix', 'Ada-SentMix', 'Ada-WordMix', 'Ada-ConceptMix']:
        transform_prob = 1.
        transform = getattr(sys.modules[__name__], t.replace('Ada-', ""))
        if hasattr(transform, 'uses_dataset'):
            transform = transform(dataset=task)
        else:
            transform = transform()
        tmcb = TargetedMixturesCallback(
            dataloader=DataLoader(eval_dataset, batch_size=4),
            device=device,
            sentence1_key=sentence1_key, 
            sentence2_key=sentence2_key,
            num_classes=num_classes
        )
        callbacks.append(tmcb)

    collator = SibylCollator( 
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        tokenize_fn=sibyl_tokenize_fn, 
        transform=transform, 
        num_sampled_INV=num_sampled_INV, 
        num_sampled_SIB=num_sampled_SIB,
        dataset=task,
        task_type=task_type, 
        tran_type=None, 
        label_type=None,
        one_hot=label_type != "hard",
        transform_prob=transform_prob,
        target_pairs=[],
        target_prob=0.5,
        reduce_mixed=False,
        num_classes=num_classes,
        return_tensors='pt',
        return_text=False,
        keep_original=keep_original
    )   

    #############################################################
    ## Trainer Setup ############################################
    #############################################################

    train_batch_size = args.train_batch_size    
    eval_batch_size = args.eval_batch_size  
    num_epoch = args.num_epoch  
    gradient_accumulation_steps = 1
    max_steps = int((len(train_dataset) * num_epoch / gradient_accumulation_steps) / train_batch_size)
    logging_steps = max_steps // num_epoch

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
    out['run_time'] = run_time
    print('ORIG for {}\n{}'.format(checkpoint, out))

    results.append(out)

    # save results
    df = pd.DataFrame(results)
    df.to_csv(save_file)

    # move best model to save_dir
    # shutil.copytree(checkpoint, save_dir)
    # shutil.rmtree(checkpoint)