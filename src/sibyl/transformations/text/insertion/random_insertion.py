from ..abstract_transformation import *
from ..tasks import *
from ..tasks import *
import numpy as np
from ..word_swap.change_synse import all_possible_synonyms

class RandomInsertion(AbstractTransformation):
    """
    Inserts random words
    """

    def __init__(self, n=1, return_metadata=False):
        """
        Initializes the transformation

        Parameters
        ----------
        n : int
            The number of random insertions to perform
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        super().__init__() 
        self.n=n
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(tran_type='SIB', label_type='hard'),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='SIB'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
    
    def __call__(self, in_text):
        new_words = in_text.split()
        if len(new_words) - 1 > 0:
            for _ in range(self.n):
                self.add_word(new_words)
            out_text = ' '.join(new_words)
        else:
            out_text = in_text
        return out_text

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

        # transform y
        if self.task_config['tran_type'] == 'INV':
            y_out = y
        else:
            soften = self.task_config['label_type'] == 'soft'
            if self.task_config['task_name'] == 'grammaticality':
                y_out = invert_label(y, soften=soften)
            elif self.task_config['task_name'] == 'similarity':
                y_out = smooth_label(y, factor=0.25)
            else:
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out
        
    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[self.np_random.integers(0, len(new_words)-1)]
            synonyms = all_possible_synonyms(random_word) #get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = self.np_random.integers(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)