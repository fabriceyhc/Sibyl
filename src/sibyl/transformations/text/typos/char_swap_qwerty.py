from ..abstract_transformation import *
import numpy as np

class RandomSwapQwerty(AbstractTransformation):
    """
    Substitues random chars
    """

    def __init__(self, task=None,meta=False):
        """
        A transformation that swaps characters with adjacent keys on a
        QWERTY keyboard, replicating the kind of errors that come from typing
        too quickly.

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.keyboard_adjacency = {
            "q": [
                "w",
                "a",
                "s",
            ],
            "w": ["q", "e", "a", "s", "d"],
            "e": ["w", "s", "d", "f", "r"],
            "r": ["e", "d", "f", "g", "t"],
            "t": ["r", "f", "g", "h", "y"],
            "y": ["t", "g", "h", "j", "u"],
            "u": ["y", "h", "j", "k", "i"],
            "i": ["u", "j", "k", "l", "o"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z", "x"],
            "s": ["q", "w", "e", "a", "d", "z", "x"],
            "d": ["w", "e", "r", "f", "c", "x", "s"],
            "f": ["e", "r", "t", "g", "v", "c", "d"],
            "g": ["r", "t", "y", "h", "b", "v", "d"],
            "h": ["t", "y", "u", "g", "j", "b", "n"],
            "j": ["y", "u", "i", "k", "m", "n", "h"],
            "k": ["u", "i", "o", "l", "m", "j"],
            "l": ["i", "o", "p", "k"],
            "z": ["a", "s", "x"],
            "x": ["s", "d", "z", "c"],
            "c": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
        }
        self.metadata = meta
    
    def __call__(self, string, n=1):
        """
        Parameters
        ----------
        word : str
            The input string
        n : int
            The number of random char Substitutions to perform

        Returns
        ----------
        ret : str
            The output with random Substitutions
        """
        original=string
        assert n <= len(string), "n is too large. n should be <= "+str(len(string))
        idx = sorted(np.random.choice(len(string), n, replace=False ))
        for i in idx:
            string = string[:i] + self.get_adjacent_letter(string[i]) + string[i + 1 :]   
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

    def get_adjacent_letter(self, s):
        s_lower = s.lower()
        if s_lower in self.keyboard_adjacency:
            adjacent_keys = self.keyboard_adjacency[s_lower]
            if s.isupper():
                ans = [key.upper() for key in adjacent_keys]
            else:
                ans = adjacent_keys
        else:
            return s
        return np.random.choice(ans)