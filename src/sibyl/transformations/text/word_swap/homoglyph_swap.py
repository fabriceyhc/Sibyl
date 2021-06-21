from ..abstract_transformation import *
import random
import numpy as np

class HomoglyphSwap(AbstractTransformation):
    """
    Transforms an input by replacing its words with 
    visually similar words using homoglyph swaps.
    """
    def __init__(self, change=0.25, task=None,meta=False):
        """
        Initializes the transformation

        Parameters
        ----------
        change: float
            tells how many of the charachters in string 
            to possibly replace
            warning: it will check change % or charachters 
            for possible replacement
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }
        assert(0<=change<=1)
        self.change = change
        self.task = task
        self.metadata = meta
    
    def __call__(self, string):
        """
        Returns a list containing all possible words with 1 character
        replaced by a homoglyph.

        Parameters
        ----------
        string : str
            The input string

        Returns
        ----------
        ret : str
            The output with random words deleted
        """

        # possibly = [k for k,j in enumerate(string) if j in self.homos]
        # indices = list(np.random.choice(possibly, int(np.ceil(self.change*len(string))), replace=False) )
                # try, catch ValueError ? safer option
        indices = np.random.choice(len(string), int(np.ceil(self.change*len(string))), replace=False)
        
        temp = string # deep copy apparently 
        for i in sorted(indices):
            if string[i] in self.homos:
                repl_letter = self.homos[string[i]]
                temp = temp[:i] + repl_letter + string[i+1:]
        assert type(temp) == str
        meta = {'change': string!=temp}
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