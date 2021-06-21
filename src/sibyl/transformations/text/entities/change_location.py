from ..abstract_transformation import *
import numpy as np
import en_core_web_sm
from ..data.locations import NAMED_ENTITIES

class ChangeLocation(AbstractTransformation):
    """
    Changes Location names
    """

    def __init__(self, task=None, meta=False):
        """
        Transforms an input by replacing names of recognized 
        location entity.

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.nlp = en_core_web_sm.load()
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
            The output with location names replaced
        """
        doc = self.nlp(string)
        newString = string
        for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
            start = e.start_char
            end = start + len(e.text)
            # print(e.text, "label is ", e.label_)
            if e.label_ in ('GPE', 'NORP', 'FAC', 'ORG', 'LOC'):
                name = newString[start:end]
                name = name.split()
                name[0] =  self._get_loc_name()
                name = " ".join(name)
                newString = newString[:start] + name + newString[end:]
        assert type(newString) == str
        meta = {'change': newString!=string}
        if self.metadata: return newString, meta
        return newString

    def _get_loc_name(self):
        """Return a random location name."""
        loc = np.random.choice(['country', 'nationality', 'city'])
        return np.random.choice(NAMED_ENTITIES[loc])

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