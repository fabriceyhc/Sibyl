from ..abstract_transformation import *
import numpy as np
import re
import spacy
import en_core_web_sm

class ChangeNumber(AbstractTransformation):
    """
    Contracts all known contractions in a string input or 
    returns the original string if none are found. 
    """

    def __init__(self, multiplier=0.2, replacement=None, task=None,meta=False):
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
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.multiplier = multiplier
        self.replacement = replacement
        self.nlp = en_core_web_sm.load()
        self.task = task
        self.metadata = meta
    
    def __call__(self, string):
        """Contracts contractions in a string (if any)

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with contractions expanded (if any)
        """
        original=string
        doc = self.nlp(string)
        nums = [x.text for x in doc if x.text.isdigit()]
        if not nums:
            meta = {'change': string!=original}
            if self.metadata: return string, meta
            return string
        for x in nums:
            # e.g. this is 4 you
            if x == '2' or x == '4':
                continue
            if self.replacement is None:
                change = int(int(x) * self.multiplier)
            else:
                change = self.replacement
            sub_re = re.compile(r'\b%s\b' % x)
            string = sub_re.sub(str(change), doc.text)
        assert type(string) == str
        meta = {'change': string!=original}
        if self.metadata: return string, meta
        return string

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['INV', 'INV'],
            'label_type': ['hard', 'hard']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df

    def transform_Xy(self, X, y):
        X_ = self(X)
        
        df = self.get_tran_types(task_name=self.task)
        tran_type = df['tran_type'].iloc[0]
        label_type = df['label_type'].iloc[0]

        if tran_type == 'INV':
            y_ = y
        elif tran_type == 'SIB':
            soften = label_type == 'hard'
            y_ = invert_label(y, soften=soften)
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_