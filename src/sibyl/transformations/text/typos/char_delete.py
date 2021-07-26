from ..abstract_transformation import *
import numpy as np
import string

class RandomCharDel(AbstractTransformation):
    """
    Deletes random chars
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, string, n=1):
        """
        Parameters
        ----------
        string : str
            The input string
        n : int
            Number of chars to be transformed

        Returns
        ----------
        string : str
            The output with random chars deleted
        """
        original = string
        idx = sorted(np.random.choice(len(string), n, replace=False ))
        for i in idx:
            string = string[:i] + string[i+1:]
        assert type(string) == str
        meta = {'change': string!=original}
        if self.metadata: return string, meta
        return string

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.task_config = [
            {
                'task_name' : 'sentiment',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'topic',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'grammaticality',
                'tran_type' : 'SIB',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'similarity',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'entailment',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'qa',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
        ]
        df = self._get_tran_types(self.task_config, task_name, tran_type, label_type)
        return df
        
    def transform_Xy(self, X, y):
        X_ = self(X)
        
        df = self.get_tran_types(task_name=self.task)
        tran_type = df['tran_type'].iloc[0]
        label_type = df['label_type'].iloc[0]

        if tran_type == 'INV':
            y_ = y
        elif tran_type == 'SIB':
            soften = label_type == 'soft'
            y_ = invert_label(y, soften=soften)
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_