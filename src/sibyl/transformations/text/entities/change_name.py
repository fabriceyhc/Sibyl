from ..abstract_transformation import *
import numpy as np
import en_core_web_sm
from ..data.persons import PERSON_NAMES

class ChangeName(AbstractTransformation):
    """
    Changes person names
    """

    def __init__(self, first_only=False,last_only=False, task=None, meta=False):
        """
        Transforms an input by replacing names of recognized name entity.

        Parameters
        ----------
        first_only : boolean
            Whether to change first name only
        last_only : boolean
            Whether to change last name only
        task : str
            the type of task you wish to transform the
            input towards
        """
        if first_only & last_only:
            raise ValueError("first_only and last_only cannot both be true")
        self.first_only = first_only
        self.last_only = last_only
        self.nlp = en_core_web_sm.load()
        self.task = task
        self.metadata = meta
    
    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The input string

        Returns
        ----------
        newString : str
            The output with person names replaced
        """
        doc = self.nlp(string)
        newString = string
        for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
            start = e.start_char
            end = start + len(e.text)
            # print(e.text, "label is ", e.label_)
            if e.label_ in ('PERSON', 'ORG'):
                name = e.text# newString[start:end]
                name = name.split()
                if len(name) == 1 or self.first_only:
                    name[0] = self._get_firstname()
                elif self.last_only:
                    name[-1] = self._get_lastname()
                elif len(name) > 1:
                    name[0], name[-1] =  self._get_firstname() , self._get_lastname()
                name = " ".join(name)
                newString = newString[:start] + name + newString[end:]
        assert type(newString) == str
        meta = {'change': newString!=string}
        if self.metadata: return newString, meta
        return newString

    def _get_lastname(self):
        """Return a random last name."""
        return np.random.choice(PERSON_NAMES["last"])

    def _get_firstname(self):
        """Return a random first name."""
        return np.random.choice(PERSON_NAMES["first"])

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