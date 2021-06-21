from ..abstract_transformation import *
import numpy as np
import random
from ..word_swap.change_synse import all_possible_synonyms

class RandomInsertion(AbstractTransformation):
    """
    Inserts random words
    """

    def __init__(self, n=1, task=None,meta=False):
        """
        Initializes the transformation

        Parameters
        ----------
        n : int
            The number of random insertions to perform
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.n=n
        self.task=task
        self.metadata = meta
    
    def __call__(self, words):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            Number of word insertions

        Returns
        ----------
        ret : str
            The output with random words inserted
        """
        new_words = words.split()
        for _ in range(self.n):
            add_word(new_words)
        ret = ' '.join(new_words)
        assert type(ret) == str
        meta = {'change': words!=ret}
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

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = all_possible_synonyms(random_word) #get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)


