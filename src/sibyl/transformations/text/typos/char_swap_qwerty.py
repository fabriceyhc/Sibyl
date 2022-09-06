from ..abstract_transformation import *
from ..tasks import *
import numpy as np

class RandomSwapQwerty(AbstractTransformation):
    """
    Substitues random chars
    """

    def __init__(self, task_name=None, return_metadata=False):
        """
        A transformation that swaps characters with adjacent keys on a
        QWERTY keyboard, replicating the kind of errors that come from typing
        too quickly.

        Parameters
        ----------
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        super().__init__(task_name) 
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
        self.task_config = self.match_task(task_name)
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
    
    def __call__(self, in_text, n=1):
        """
        Parameters
        ----------
        in_text : str
            The input string
        n : int
            The number of random char Substitutions to perform

        Returns
        ----------
        in_text : str
            The output with random Substitutions
        """
        if len(in_text) > 0:
            assert n <= len(in_text), "n is too large. n should be <= "+str(len(in_text))
            idx = sorted(self.np_random.choice(len(in_text), n, replace=False))
            for i in idx:
                in_text = in_text[:i] + self.get_adjacent_letter(in_text[i]) + in_text[i + 1 :]   
        return in_text

    def get_task_configs(self, task_name=None, tran_type=None, label_type=None):
        init_configs = [task() for task in self.task_configs]
        df = self._get_task_configs(init_configs, task_name, tran_type, label_type)
        return df

    def transform_Xy(self, X, y):

        # transform X
        if isinstance(X, str):
            X = [X]

        assert len(X) == len(self.task_config['input_idx']), ("The number of inputs does not match the expected "
                                                         "amount of {} for the {} task".format(
                                                            self.task_config['input_idx'],
                                                            self.task_config['task_name']))

        X_out = []
        for i, x in zip(self.task_config['input_idx'], X):
            if i == 0:
                X_out.append(x)
                continue
            X_out.append(self(x))

        metadata = {'change': X != X_out}
        X_out = X_out[0] if len(X_out) == 1 else X_out

        y_out = y
        if metadata['change']:
            # transform y
            if self.task_config['tran_type'] == 'INV':
                y_out = y
            else:
                soften = self.task_config['label_type'] == 'soft'
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

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
        return self.np_random.choice(ans)