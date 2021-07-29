from ..abstract_transformation import *
from ..tasks import *
from ..data.persons import PERSON_NAMES
import numpy as np
import en_core_web_sm

class ChangeName(AbstractTransformation):
    """
    Changes person names
    """

    def __init__(self, first_only=False, last_only=False, return_metadata=False):
        """
        Transforms an input by replacing names of recognized name entity.

        Parameters
        ----------
        first_only : boolean
            Whether to change first name only
        last_only : boolean
            Whether to change last name only
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        if first_only & last_only:
            raise ValueError("first_only and last_only cannot both be true")
        self.first_only = first_only
        self.last_only = last_only
        self.nlp = en_core_web_sm.load()
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='SIB'),
            Entailment(input_idx=[1,0], tran_type='SIB'),
            Entailment(input_idx=[0,1], tran_type='SIB'),
            Entailment(input_idx=[1,1], tran_type='SIB'),
        ]
    
    def __call__(self, in_text):
        doc = self.nlp(in_text)
        out_text = in_text
        for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
            start = e.start_char
            end = start + len(e.text)
            # print(e.text, "label is ", e.label_)
            if e.label_ in ('PERSON', 'ORG'):
                name = e.text# out_text[start:end]
                name = name.split()
                if len(name) == 1 or self.first_only:
                    name[0] = self._get_firstname()
                elif self.last_only:
                    name[-1] = self._get_lastname()
                elif len(name) > 1:
                    name[0], name[-1] =  self._get_firstname() , self._get_lastname()
                name = " ".join(name)
                out_text = out_text[:start] + name + out_text[end:]
        return out_text

    def _get_lastname(self):
        """Return a random last name."""
        return np.random.choice(PERSON_NAMES["last"])

    def _get_firstname(self):
        """Return a random first name."""
        return np.random.choice(PERSON_NAMES["first"])

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
            if task_config['task_name'] == 'similarity':
                # hard code for now... :(
                # 0 = dissimilar, 1 = similar
                if isinstance(y, int):
                    if y == 0:
                        y_out = 0
                    else:
                        y_out = invert_label(y, soften=soften)
                else:
                    if np.argmax(y) == 0:
                        y_out = 0
                    else:
                        y_out = smooth_label(y, factor=0.25)
            elif task_config['task_name'] == 'entailment':
                # hard coded for now... :(
                # 0 = entailed, 1 = neutral, 2 = contradiction
                if isinstance(y, int):
                    if y in [0, 2]:
                        y_out = 1
                    else: 
                        y_out = y
                else:
                    if np.argmax(y) in [0, 2]:
                        y_out = 1
                    else:
                        y_out = y
            else:
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out