from ..abstract_transformation import *
from ..tasks import *
from ..word_swap.change_synse import ChangeAntonym

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import pke
import random
import itertools
import string
 
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    AutoModelForSequenceClassification,
)
from huggingface_hub import HfApi
from .sequence_classification import SequenceClassificationExplainer

class Concept2Sentence(AbstractTransformation):
    """
    Accepts an input, extracts salient keywords, and 
    generates a new sentence using all or most of the
    extracted keywords. This is intended to generate 
    comparable text for NLP tasks like sentiment 
    analysis, topic classification, etc. 
    """

    uses_dataset = True
    
    def __init__(self, 
                 return_metadata=False, 
                 dataset=None, 
                 extract="token",
                 gen_beam_size=10,
                 text_min_length=10,
                 text_max_length=128,
                 device='cuda',
                 antonymize=False,
                 require_concepts_in_new_text=False,
                 lemmatizer=None,
                 return_concepts=False):
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='SIB'),
            Entailment(input_idx=[1,0], tran_type='SIB'),
            Entailment(input_idx=[0,1], tran_type='SIB'),
            Entailment(input_idx=[1,1], tran_type='SIB'),
        ]
        self.dataset = dataset
        self.extract = extract
        self.gen_beam_size = gen_beam_size
        self.text_min_length = text_min_length
        self.text_max_length = text_max_length
        self.device = torch.device('cuda') if torch.cuda.is_available() and 'cuda' in device else torch.device('cpu')
        self.antonymize = antonymize
        self.return_concepts = return_concepts
        self.extractor = RationalizedKeyphraseExtractor(dataset=self.dataset, 
                                                        extract=self.extract, 
                                                        device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("sibyl/BART-commongen")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("sibyl/BART-commongen").to(self.device)
        self.lemmatizer = WordNetLemmatizer().lemmatize if lemmatizer is None else lemmatizer
        self.require_concepts_in_new_text = require_concepts_in_new_text
        if self.antonymize:
            self.antonymizer = ChangeAntonym()
    
    def __call__(self, in_text, in_target=None, n=None, threshold=None):
        concepts = self.extract_concepts(in_text, in_target, n, threshold)
        if concepts:
            new_sentence = self.generate_text_from_concepts(concepts)
            if self.require_concepts_in_new_text:
                text_lemmas = self.lemmatizer(new_sentence)
                used_concepts = [c for c in concepts if self.lemmatizer(c) in text_lemmas]
                if concepts != used_concepts:
                    while concepts != used_concepts and len(concepts) > 1:
                        concepts = concepts[:-1]
                        new_text = self.generate_text_from_concepts(concepts)
                        text_lemmas = self.lemmatizer(new_text)
                        used_concepts = [c for c in concepts if self.lemmatizer(c) in text_lemmas]
                    if concepts == used_concepts:
                        new_sentence = new_text
        else:
            new_sentence = in_text
            concepts = []
        if self.return_concepts:
            return concepts, new_sentence
        return new_sentence

    def extract_concepts(self, in_text, in_target, n=None, threshold=None):
        if not isinstance(in_target, int):
            in_target = np.argmax(in_target)                
        # extract concepts
        concepts = self.extractor(in_text, label_idx=in_target, n=n, threshold=threshold)
        if not concepts:
            concepts = RationalizedKeyphraseExtractor(extract="concepts")(in_text, label_idx=in_target, n=2)
        if self.antonymize:
            concepts = list(set([self.antonymizer(c) for c in concepts]))
        return concepts

    def generate_text_from_concepts(self, concepts):
        # prepare batch --> generate sentence from concepts
        batch = {'concepts': [concepts], 'target': ''}
        new_sentences = beam_generate_sentences(
            batch,
            self.model,
            self.tokenizer,
            num_beams=self.gen_beam_size,
            min_length=self.text_min_length,
            max_length=self.text_max_length,
            device=self.device
        )
        if len(new_sentences) == 1:
            return new_sentences[0]
        return new_sentences

    def get_task_configs(self, task_name=None, tran_type=None, label_type=None):
        init_configs = [task() for task in self.task_configs]
        df = self._get_task_configs(init_configs, task_name, tran_type, label_type)
        return df

    def transform_Xy(self, X, y, task_config):

        orig_return_concepts_state = self.return_concepts
        self.return_concepts = True

        # transform X
        if isinstance(X, str):
            X = [X]

        assert len(X) == len(task_config['input_idx']), ("The number of inputs does not match the expected "
                                                         "amount of {} for the {} task".format(
                                                            task_config['input_idx'],
                                                            task_config['task_name']))

        X_out = []
        concepts_out =[]
        for i, x in zip(task_config['input_idx'], X):
            if i == 0:
                X_out.append(x)
                concepts_out.append([])
                continue
            concepts, new_sentence = self(x, y)
            X_out.append(new_sentence)
            concepts_out.append(concepts)

        metadata = {'change': X != X_out}
        X_out = X_out[0] if len(X_out) == 1 else X_out

        # transform y
        if task_config['tran_type'] == 'INV':
            y_out = y
        else:                
            soften = task_config['label_type'] == 'soft'
            if task_config['task_name'] == 'similarity':
                # hard code for now... :(
                # 0 = dissimilar, 1 = similar
                if (task_config['input_idx'] == [1,1] and concepts_out[0] == concepts_out[-1]):
                        y_out = y
                else:
                    if isinstance(y, int):
                        if y == 0:
                            y_out = 0
                        else:
                            y_out = invert_label(y, soften=soften)
                    else:
                        if np.argmax(y) == 0:
                            y_out = 0
                        else:
                            y_out = smooth_label(y, factor=0.25)
            elif task_config['task_name'] == 'entailment':
                # hard coded for now... :(
                # 0 = entailed, 1 = neutral, 2 = contradiction
                if isinstance(y, int):
                    if y in [0, 2]:
                        y_out = 1
                    else: 
                        y_out = y
                else:
                    if np.argmax(y) in [0, 2]:
                        y_out = 1
                    else:
                        y_out = y
            else:
                y_out = invert_label(y, soften=soften)

        self.return_concepts = orig_return_concepts_state
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

##########################################
## Keyword/Keyphrase Extraction Helpers ##
##########################################

def extract_ngrams(in_text, 
                   grams=3, 
                   return_lesser_grams=True, 
                   remove_stopwords=True):
  
    stops = []
    if remove_stopwords:
        stops = stopwords.words('english')
        stops.remove('not')

    def _get_ngram_phrases(in_text, gram):
        all_grams = []
        n_out = ngrams(in_text.split(), gram)
        for out in n_out:
            # if not all([w in stops for w in out]):
            if sum([w not in stops for w in out]) / len(out) > 0.5:
                out = ' '.join(list(out))
                all_grams.append(out) 
        return all_grams
    
    #strip punctuation
    in_text = in_text.translate(str.maketrans('', '', string.punctuation))

    if return_lesser_grams:
        all_grams = []
        for gram in range(1,grams+1):
            all_grams.extend(_get_ngram_phrases(in_text, gram))
    else:
        all_grams = _get_ngram_phrases(in_text, grams)
    return [all_grams]

def extract_keyphrases(in_text, n=None):
    if n is None:
        n = random.randint(2,5)
 
    # extract key phrases
    pos = {'NOUN', 'PROPN', 'ADJ', 'VERB'}
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=in_text, language='en')
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=n)
    keyphrases = [p[0] for p in keyphrases]
    return [keyphrases]
 
def extract_concepts(in_text, n=None):
    if n is None:
        n = random.randint(2,5)
 
    # extract key phrases
    keyphrases = extract_keyphrases(in_text, n)[0]
 
    # distill phrases to concepts
    concepts = set([kp if len(kp.split()) == 1 
                    else random.sample(kp.split(),1)[0] for kp in keyphrases])
    return [list(concepts)]

class RationalizedKeyphraseExtractor:
    def __init__(self, 
                 dataset=None, 
                 device='cpu', 
                 extract='tokens',
                 randomize=True,
                 remove_stopwords=True,
                 lemmatizer = None):
      
        self.dataset = dataset
        self.device = device
        self.extract = extract
        self.randomize = randomize
        self.remove_stopwords = remove_stopwords
        self.model = None
        self.tokenizer = None
        self.interpreter = None
        self.lemmatizer = WordNetLemmatizer().lemmatize if lemmatizer is None else lemmatizer

        if dataset is not None:
            api = HfApi()
            # find huggingface model to provide rationalized output
            modelIds = api.list_models(filter=("pytorch", "dataset:" + dataset, "sibyl"))
            if modelIds:
                modelId = getattr(modelIds[0], 'modelId')
                print('Using ' + modelId + ' to rationalize keyphrase selections.')
                self.model = AutoModelForSequenceClassification.from_pretrained(modelId).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(modelId)
                self.interpreter = SequenceClassificationExplainer(self.model, self.tokenizer)
        else:
            self.extract = "concepts"
        self.stops = stopwords.words('english') if self.remove_stopwords else []

    def __call__(self, in_text, label_idx=None, n=None, threshold=None): 

        if self.randomize and n is None:
            n = np.random.randint(1,5)     

        if 'keyphrase' in self.extract:
            keyphrases = extract_keyphrases(in_text, n=10)[0]
        elif 'ngram' in self.extract:
            keyphrases = extract_ngrams(in_text, 
                            remove_stopwords=self.remove_stopwords)[0]
        elif 'concept' in self.extract:
            keyphrases = extract_concepts(in_text, n=10)[0]
        elif 'token' in self.extract:
            keyphrases = word_tokenize(in_text)
        else:
            keyphrases = in_text.split()
    
        if self.interpreter is not None: 
            if self.randomize and threshold is None:
                threshold = np.random.uniform(0.5, 0.75)

            attributions = self.interpreter(text=in_text, index=label_idx, internal_batch_size=1)
            tokens, weights = zip(*attributions)
            words, weights = merge_bpe(tokens, weights)
            if threshold is not None: 
                weights = scale(torch.from_numpy(np.array(weights).copy())).tolist()
            attributions = list(zip(words, weights))

            keyphrase_attributions = []
            for keyphrase in keyphrases:
                keyphrase_weight = 0
                keyphrase_tokens = keyphrase.split()
                for token in keyphrase_tokens:
                    for attr_pair in attributions:
                        attr, weight = attr_pair
                        if token.lower() == attr.lower() and token.lower() not in self.stops:
                            keyphrase_weight += weight 
                keyphrase_attributions.append((keyphrase, keyphrase_weight / len(keyphrase_tokens)))
            keyphrase_attributions.sort(key = lambda x: x[-1], reverse=True) 
            if threshold is not None:
                concepts = list(set([x[0].lower() for x in keyphrase_attributions if x[1] >= threshold]))
            else: 
                concepts = list(set([k[0].lower() for k in keyphrase_attributions[:n]]))
        else:
            concepts = list(set([x.lower() for x in keyphrases[:n]]))

        # cleanup concepts ############################################
        
        # remove punctuations + single characters
        concept_depunct = [c.translate(str.maketrans('', '', string.punctuation)) for c in concepts]
        concepts = [c1 for c1, c2 in zip(concepts, concept_depunct) if len(c2) > 1]
        
        # remove matching lemmas
        concept_lemmas = [self.lemmatizer(c) for c in concepts]
        concepts = [c for i, c in enumerate(concepts) if self.lemmatizer(c) not in concept_lemmas[:i]]

        return concepts

###########################
## Miscellaneous Helpers ##
###########################

def scale(out, rmax=1, rmin=0):
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

def merge_bpe(tok, boe, chars="##"):
    len_chars = len(chars)

    new_tok = []
    new_boe = []

    emb = []
    append = ""
    for t, e in zip(tok[::-1], boe[::-1]):
        t += append
        emb.append(e)
        if t.startswith(chars):
            append = t[len_chars:]
        else:
            append = ""
            new_tok.append(t)
            new_boe.append(np.stack(emb).mean(axis=0))
            emb = []  
    new_tok = np.array(new_tok)[::-1]
    new_boe = np.array(new_boe)[::-1]
    return new_tok, new_boe

#############################
## Text Generation Helpers ##
#############################

def construct_input_for_batch(batch):
    """
    Function that takes a batch from a dataset and constructs the corresponding
    input string.
    """
    source = [' '.join(concepts) for concepts in batch["concepts"]]
    target = batch["target"]
    return source, target
 
def make_batch_inputs(batch, tokenizer, device='cuda:0'):
    """
    Function that takes a batch from a dataset and formats it as input to model.
    """
    # Concatenate the concept names for each example in the batch.
    input_lists, _ = construct_input_for_batch(batch)
    # Use the model's tokenizer to create the batch input_ids.
    batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
    # Move all inputs to the device.
    batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
    return batch_features
 
def batch_tokenize(dataset_batch, tokenizer, decoder_max_length=32):
    """
    Construct the batch (source, target) and run them through a tokenizer.
    """
    source, target = construct_input_for_batch(dataset_batch)
    res = {
        "input_ids": tokenizer(source)["input_ids"],
        "labels": tokenizer(
            target,
            padding='max_length',
            truncation=True,
            max_length=decoder_max_length
        )["input_ids"],
    }
    return res
 
def beam_generate_sentences(batch,
                            model,
                            tokenizer,
                            num_beams=4,
                            min_length=10,
                            max_length=32,
                            device='cuda:0'):
    """
    Function to generate outputs from a model with beam search decoding.
    """
    # Create batch inputs.
    features = make_batch_inputs(
        batch=batch,
        tokenizer=tokenizer,
        device=device)
    
    # Generate with beam search.
    generated_ids = model.generate(
        input_ids=features['input_ids'],
        attention_mask=features['attention_mask'],
        num_beams=num_beams,
        min_length=min_length,
        max_length=max_length,
    ).cpu().detach()
    # Use model tokenizer to decode to text.
    generated_sentences = [
        tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        for gen_ids in generated_ids
    ]
    return generated_sentences

def visualize_salience(in_text, target, dataset):
    modelIds = api.list_models(filter=("pytorch", "dataset:" + dataset), sort='downloads', direction=-1)
    if modelIds:
        modelId = getattr(modelIds[0], 'modelId')
        print('Using ' + modelId + ' to rationalize keyphrase selections.')

    interp_model = AutoModelForSequenceClassification.from_pretrained(modelId).to(device)
    interp_tokenizer = AutoTokenizer.from_pretrained(modelId)

    cls_explainer = SequenceClassificationExplainer(interp_model, interp_tokenizer)
    attributions = cls_explainer(in_text, target)
    print(cls_explainer.predicted_class_name)
    cls_explainer.visualize()
    return attributions