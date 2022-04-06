import os
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoModel, 
    AutoTokenizer
)
from datasets import load_dataset, load_from_disk

import matplotlib.pyplot as plt
import umap.umap_ as umap
import umap.plot

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=250)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def encode(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)
    emb = out[0]
    return emb

def flatten_sib(labels, sib_val=0.5):
    new_label = []
    for i in range(len(labels)):
        if type(labels[i]) == int:
            new_label.append(labels[i])
        elif any([x>0 and x<1 for x in labels[i]]):
                new_label.append(sib_val)
        else:
            new_label.append(np.array(labels[i]).argmax())   
    return new_label    

parser = argparse.ArgumentParser(description='UMAP Visualizer')

parser.add_argument('--data-dir', type=str, default="../../data1/fabricehc/prepared_datasets/",
                    help='path to data folders')
parser.add_argument('--save-dir', type=str, default="./imgs/",
                    help='path to save vizualizations')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='encoding batchsize')
parser.add_argument('--gpus', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--datasets_cache', default="../../data1/fabricehc/.cache", type=str,
                    help='location for for HF_DATASETS_CACHE')
parser.add_argument('--transforms', nargs='+', 
                    default=['ORIG', 'INV', 'SIB', 'INVSIB', 'Concept2Sentence', 
                             'TextMix', 'SentMix', 'WordMix',  'ConceptMix'], 
                    type=str, help='transformed datasets to search for')
parser.add_argument('--tasks', nargs='+', 
                    default=['amazon_polarity', 'yahoo_answers_topics', 
                             'dbpedia_14', 'imdb', 'yelp_polarity', 'ag_news'], 
                    type=str, help='tasks (datasets) to visualize, from prepared_datasets')
parser.add_argument('--num-train-per-class', nargs='+', default=[10, 200, 2500], type=int, 
                    help='number of training examples per class')
parser.add_argument('--models', nargs='+',  default=['bert-base-uncased'], 
                    type=str, help='pretrained huggingface models to train')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['HF_DATASETS_CACHE'] = args.datasets_cache

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir = args.data_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

batch_size = args.batch_size

num_train_examples = args.num_train_per_class
MODEL_NAMES = args.models
ts = args.transforms
tasks = args.tasks

run_args = []
for num_train_per_class in num_train_examples:
    for t in ts:   
        for task in tasks:
            for MODEL_NAME in MODEL_NAMES:
                run_args.append({
                    "num_train_per_class":num_train_per_class,
                    "task":task,
                    "MODEL_NAME":MODEL_NAME,
                    "t":t
                })

for run_arg in run_args:
    
    print(pd.DataFrame([run_arg]))
    
    num_train_per_class = run_arg['num_train_per_class']
    task = run_arg['task']
    MODEL_NAME = run_arg['MODEL_NAME']
    t = run_arg['t']

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
    MODEL_NAME = 'fabriceyhc/' + MODEL_NAME + '-' + task
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, num_labels=num_classes).to(device)

    data_path = os.path.join(data_dir, task, t, task + '_train_' + str(num_train_per_class))
    if not os.path.exists(data_path):
    	continue

    dataset = load_from_disk(data_path)
    print(dataset)
    
    inputs = tokenizer(dataset['text'], return_tensors='pt', padding=True, truncation=True)
    batches = zip(chunker(inputs.input_ids, batch_size), 
                  chunker(inputs.attention_mask, batch_size))
    
    i = 0
    cls_tokens = np.empty((len(dataset['text']), model.config.hidden_size))
    for input_ids, attention_mask in tqdm(batches, total=len(dataset['text'])//batch_size):
        emb = encode(model, input_ids.cuda(), attention_mask.cuda()).cpu().numpy()[:, 0, :]
        cls_tokens[i:i+batch_size] = emb
        i += batch_size
    mapper = umap.UMAP().fit(cls_tokens)
    umap.plot.points(mapper, labels=np.array(flatten_sib(dataset['label'])), theme='fire')
    plt.title('UMAP | ' + task + ' | ' + t)
    plt.savefig(os.path.join(save_dir, 'umap' + '_' + task + '_' + t + '_' + str(num_train_per_class) + '.png'))
    plt.show()