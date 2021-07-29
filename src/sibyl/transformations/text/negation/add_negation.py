from ..abstract_transformation import *
from ..tasks import *
import collections
import pattern
from pattern import en
import spacy
import en_core_web_sm
import numpy as np

class AddNegation(AbstractTransformation):
    """
    Defines a transformation that negates a string.
    """

    def __init__(self, return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.nlp = en_core_web_sm.load()
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='SIB'),
            Entailment(input_idx=[0,1], tran_type='SIB'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
    
    def __call__(self, in_text):
        ans = self.wrapper(in_text)
        if ans is None: ans = in_text
        return ans

    def wrapper(self, in_text):
        doc = self.nlp(in_text)
        doc_len = len(doc)
        for sentence in doc.sents:
            if len(sentence) < 3:
                continue
            root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
            root = doc[root_id]
            if '?' in sentence.text and sentence[0].text.lower() == 'how':
                continue
            if root.lemma_.lower() in ['thank', 'use']:
                continue
            if root.pos_ not in ['VERB', 'AUX']:
                continue
            neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
            if neg:
                continue
            if root.lemma_ == 'be':
                if '?' in sentence.text:
                    continue
                if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
                    if root_id + 1 >= doc_len:
                        out_text = in_text
                    else:
                        out_text = doc[:root_id + 1].text + ' not ' + doc[root_id + 1:].text
                    return out_text
                else:
                    if root_id == 0:
                        out_text = doc[root_id].text + ' not ' + doc[root_id:].text
                    else:    
                        out_text = doc[:root_id].text + ' not ' + doc[root_id:].text
                    return out_text
            else:
                aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
                if aux:
                    aux = aux[0]
                    if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
                        lemma = doc[aux.i].lemma_.lower()
                        if lemma == 'will':
                            fixed = 'won\'t'
                        elif lemma == 'have' and doc[aux.i].text in ['\'ve', '\'d']:
                            fixed = 'haven\'t' if doc[aux.i].text == '\'ve' else 'hadn\'t'
                        elif lemma == 'would' and doc[aux.i].text in ['\'d']:
                            fixed = 'wouldn\'t'
                        else:
                            fixed = doc[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
                        fixed = ' %s ' % fixed
                        if aux.i == 0:
                            out_text = doc[aux.i].text + fixed + doc[aux.i + 1:].text
                        else:
                            out_text = doc[:aux.i].text + fixed + doc[aux.i + 1:].text
                        return out_text
                    if root_id  == 0:
                        out_text = doc[root_id].text + ' not ' + doc[root_id:].text
                    else:
                        out_text = doc[:root_id].text + ' not ' + doc[root_id:].text
                    return out_text
                else:
                    # TODO: does, do, etc. 
                    subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
                    p = pattern.en.tenses(root.text)
                    tenses = collections.Counter([x[0] for x in pattern.en.tenses(root.text)]).most_common(1)
                    tense = tenses[0][0] if len(tenses) else 'present'
                    params = [tense, 3]
                    if p:
                        tmp = [x for x in p if x[0] == tense]
                        if tmp:
                            params = list(tmp[0])
                        else:
                            params = list(p[0])
                    if root.tag_ not in ['VBG']:
                        do = pattern.en.conjugate('do', *params) + 'n\'t'
                        new_root = pattern.en.conjugate(root.text, tense='infinitive')
                    else:
                        do = 'not'
                        new_root = root.text
                    # print(doc)
                    # print(len(doc))
                    # print(root_id)
                    # print(do)
                    # print(new_root)
                    # print(doc[root_id + 1:].text)
                    if root_id == 0 or root_id + 1 >= doc_len:
                        out_text = in_text
                    else:
                        out_text = '%s %s %s %s' % (doc[:root_id].text, do, new_root, doc[root_id + 1:].text)
                    return out_text

    def get_task_configs(self, task_name=None, tran_type=None, label_type=None):
        init_configs = [task() for task in self.task_configs]
        df = self._get_task_configs(init_configs, task_name, tran_type, label_type)
        return df
        
    def transform_Xy(self, X, y, task_config):

        # transform X
        if isinstance(X, str):
            X = [X]

        assert len(X) == len(task_config['input_idx']), ("The number of inputs does not match the expected "
                                                         "amount of {} for the {} task".format(
                                                            task_config['input_idx'],
                                                            task_config['task_name']))

        X_out = []
        for i, x in zip(task_config['input_idx'], X):
            if i == 0:
                X_out.append(x)
                continue
            X_out.append(self(x))

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
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out