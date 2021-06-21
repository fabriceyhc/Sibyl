from ..abstract_transformation import *
import numpy as np
import string

class RandomCharSubst(AbstractTransformation):
    """
    Substitues random chars
    """
    def __init__(self, task=None,meta=False):
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
    
    def __call__(self, text, n=1):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            The number of random Substitutions to perform

        Returns
        ----------
        ret : str
            The output with random Substitutions
        """
        assert n <= len(text), "n is too large. n should be <= "+str(len(text))
        idx = sorted(np.random.choice(len(text), n, replace=False ))
        temp = text
        for i in idx:
            temp = temp[:i] + get_random_letter() + temp[i + 1 :]
        meta = {'change': temp!=text}
        if self.metadata: return temp, meta
        return temp

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

def get_random_letter():
    # printable = digits + ascii_letters + punctuation + whitespace
    return np.random.choice(list(string.printable))

