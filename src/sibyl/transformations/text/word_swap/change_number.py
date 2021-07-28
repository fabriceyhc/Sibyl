from ..abstract_transformation import *
from ..tasks import *
import numpy as np
import re
import spacy
import en_core_web_sm

class ChangeNumber(AbstractTransformation):
    """
    Contracts all known contractions in a string input or 
    returns the original string if none are found. 
    """

    def __init__(self, multiplier=0.2, replacement=None, return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        multiplier : float
            The value by which all numbers in the input
            string will be multiplied
        repalcement : float
            The value by which all numbers in the input
            string will be replaced (overrides multiplier)
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        self.multiplier = multiplier
        self.replacement = replacement
        self.nlp = en_core_web_sm.load()
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
    
    def __call__(self, in_text):
        doc = self.nlp(in_text)
        nums = [x.text for x in doc if x.text.isdigit()]
        if not nums:
            return in_text
        for x in nums:
            # e.g. this is 4 you
            if x == '2' or x == '4':
                continue
            if self.replacement is None:
                change = int(int(x) * self.multiplier)
            else:
                change = self.replacement
            sub_re = re.compile(r'\b%s\b' % x)
            out_text = sub_re.sub(str(change), doc.text)
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