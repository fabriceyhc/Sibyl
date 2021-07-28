from ..abstract_transformation import *
from ..tasks import *
import random
import numpy as np

class HomoglyphSwap(AbstractTransformation):
    """
    Transforms an input by replacing its words with 
    visually similar words using homoglyph swaps.
    """
    def __init__(self, change=0.25, return_metadata=False):
        """
        Initializes the transformation

        Parameters
        ----------
        change: float
            tells how many of the charachters in string 
            to possibly replace
            warning: it will check change % or charachters 
            for possible replacement
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
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
        assert 0<=change<=1, "Change must be a probability between 0 and 1"
        self.change = change
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
    
    def __call__(self, in_text):
        """
        Returns a list containing all possible words with 1 character
        replaced by a homoglyph.

        Parameters
        ----------
        in_text : str
            The input string

        Returns
        ----------
        out_text : str
            The output with random words deleted
        """

        # possibly = [k for k,j in enumerate(in_text) if j in self.homos]
        # indices = list(np.random.choice(possibly, int(np.ceil(self.change*len(in_text))), replace=False) )
                # try, catch ValueError ? safer option
        indices = np.random.choice(len(in_text), int(np.ceil(self.change*len(in_text))), replace=False)
        
        out_text = in_text # deep copy apparently 
        for i in sorted(indices):
            if in_text[i] in self.homos:
                repl_letter = self.homos[in_text[i]]
                out_text = out_text[:i] + repl_letter + in_text[i+1:]
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