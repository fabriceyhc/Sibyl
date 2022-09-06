from ..abstract_transformation import *
from ..tasks import *
import numpy as np
import string

class RandomCharSwap(AbstractTransformation):
    """
    Swaps random consecutive chars
    """
    def __init__(self, task_name=None, return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
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
        
    def __call__(self, in_text, n=1):
        """
        Parameters
        ----------
        in_text : str
            The input string
        n : int
            Number of chars to be transformed

        Returns
        ----------
        in_text : str
            The output with random chars pairs swapped
        """
        if len(in_text)-1 > 0:
            idx = sorted(self.np_random.choice(len(in_text)-1, n, replace=False))
            for i in idx:
                in_text = in_text[:i] + in_text[i + 1] + in_text[i] + in_text[i + 2 :]
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