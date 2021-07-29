from ..abstract_transformation import *
from ..tasks import *
import collections
import pattern
import spacy
import en_core_web_sm
import numpy as np

class RemoveNegation(AbstractTransformation):
    """
    Defines a transformation that removes a negation
    if found within the input. Otherwise, return the 
    original string unchanged. 
    """

    def __init__(self, return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
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
        # This removes all negations in the doc. I should maybe add an option to remove just some.
        doc = self.nlp(in_text)
        doc_len = len(doc)
        notzs = [i for i, z in enumerate(doc) if z.lemma_ == 'not' or z.dep_ == 'neg']
        new = []
        for notz in notzs:
            before = doc[notz - 1] if notz != 0 else None
            after = doc[notz + 1] if len(doc) > notz + 1 else None
            if (after and after.pos_ == 'PUNCT') or (before and before.text in ['or']):
                continue
            new.append(notz)
        notzs = new
        if not notzs:
            return in_text
        out_text = ''
        start = 0
        for i, notz in enumerate(notzs):
            id_start = notz
            to_add = ' '
            id_end = notz + 1
            before = doc[notz - 1] if notz != 0 else None
            after = doc[notz + 1] if len(doc) > notz + 1 else None
            if before and before.lemma_ in ['will', 'can', 'do']:
                id_start = notz - 1
                tenses = collections.Counter([x[0] for x in pattern.en.tenses(before.text)]).most_common(1)
                tense = tenses[0][0] if len(tenses) else 'present'
                p = pattern.en.tenses(before.text)
                params = [tense, 3]
                if p:
                    tmp = [x for x in p if x[0] == tense]
                    if tmp:
                        params = list(tmp[0])
                    else:
                        params = list(p[0])
                to_add = ' '+ pattern.en.conjugate(before.lemma_, *params) + ' '
            if before and after and before.lemma_ == 'do' and after.pos_ == 'VERB':
                id_start = notz - 1
                tenses = collections.Counter([x[0] for x in pattern.en.tenses(before.text)]).most_common(1)
                tense = tenses[0][0] if len(tenses) else 'present'
                p = pattern.en.tenses(before.text)
                params = [tense, 3]
                if p:
                    tmp = [x for x in p if x[0] == tense]
                    if tmp:
                        params = list(tmp[0])
                    else:
                        params = list(p[0])
                to_add = ' '+ pattern.en.conjugate(after.text, *params) + ' '
                id_end = notz + 2
            # print(doc)
            # print(start)
            # print(id_start)
            # print(to_add)
            if start == id_start or start >= doc_len or id_start >= doc_len:
                continue
            else:
                out_text += doc[start:id_start].text + to_add
            start = id_end
        if id_end < doc_len:
            out_text += doc[id_end:].text
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