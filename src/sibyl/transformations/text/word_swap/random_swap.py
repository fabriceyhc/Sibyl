from ..abstract_transformation import *
import numpy as np
import random
import re

class RandomSwap(AbstractTransformation):
    """
    Swaps random words
    """

    def __init__(self, n=1, task=None,meta=False):
        """
        Initializes the transformation

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.n=n
        self.task=task
        self.metadata = meta
    
    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The input string
        n : int
            Number of word swaps

        Returns
        ----------
        ret : str
            The output with random words swapped
        """
        new_words = string.split()
        for _ in range(self.n):
            new_words = swap_word(new_words)
        ret = ' '.join(new_words)
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

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

def swap_word(new_words):
    if len(new_words)-1 <= 0:
        return new_words
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words