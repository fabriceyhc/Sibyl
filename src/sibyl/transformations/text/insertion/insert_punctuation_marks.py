from ..abstract_transformation import *
from ..tasks import *
import numpy as np
import random

class InsertPunctuationMarks(AbstractTransformation):
    """
    Inserts random punctuation marks to random spots following the 
    implementation from AEDA:
        - https://arxiv.org/abs/2108.13230
        - https://github.com/akkarimi/aeda_nlp/blob/master/code/aeda.py

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
        self.PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
        
    def __call__(self, in_text, punc_ratio = 0.3):
        """
        Parameters
        ----------
        in_text : str
            The input string
        punc_ratio : int
            Ratio of punctuation marks to words in the transformed in_text

        Returns
        ----------
        in_text : str
            The output with random chars inserted
        """
        if len(in_text) == 0:
            return in_text
        words = in_text.split(' ')
        out_text = []
        q = random.randint(1, int(punc_ratio * len(words) + 1))
        qs = random.sample(range(0, len(words)), q)

        for j, word in enumerate(words):
            if j in qs:
                out_text.append(self.PUNCTUATIONS[random.randint(0, len(self.PUNCTUATIONS)-1)])
                out_text.append(word)
            else:
                out_text.append(word)
        out_text = ' '.join(out_text)
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
            y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out