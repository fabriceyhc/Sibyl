from ..abstract_transformation import *
from ..tasks import *
import pandas as pd
import numpy as np
import os

class AddSentimentLink(AbstractTransformation):
    """
    Appends a given / constructed URL to a string input.
    Current implementation constructs a default URL that
    makes use of dictionary.com and is sensitive to changes
    in routing structure. 
    """

    def __init__(self, url=None, sentiment='positive', return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        Parameters
        ----------
        url : str
            The URL to append to an input string
        sentiment : str
            Determines whether the constructed URL will 
            feature a positive or negative sentiment.
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        
        self.url = url
        if self.url is None:
            self.url = 'https://www.dictionary.com/browse/'
            self.default_url = True
        else:
            self.default_url = False
        self.sentiment = sentiment
        if self.sentiment.lower() not in ['positive', 'negative']:
            raise ValueError("Sentiment must be 'positive' or 'negative'.")
        # https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
        cur_path = os.path.dirname(os.path.realpath(__file__))
        pos_path = os.path.join(cur_path, '../data/opinion-lexicon-English/positive-words.txt')
        neg_path = os.path.join(cur_path, '../data/opinion-lexicon-English/negative-words.txt')
        self.pos_words = pd.read_csv(pos_path, skiprows=30, names=['word'], encoding='latin-1')
        self.neg_words = pd.read_csv(neg_path, skiprows=30, names=['word'], encoding='latin-1')
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
    
    def __call__(self, in_text):
        if self.default_url:
            if 'positive' in self.sentiment:
                word = self.pos_words.sample(1)['word'].iloc[0]
            if 'negative' in self.sentiment:
                word = self.neg_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        out_text = in_text + ' ' + link
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
            elif task_config['task_name'] == 'sentiment':
                if self.sentiment == 'positive':
                    y_out = smooth_label(y, factor=0.5)
                if self.sentiment == 'negative':
                    y_out = smooth_label(y, factor=0.5)
            else:
                soften = task_config['label_type'] == 'soft'
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class AddPositiveLink(AddSentimentLink):
    def __init__(self, return_metadata=False):
        super().__init__(url=None, sentiment='positive', return_metadata=False)
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]

    def __call__(self, in_text):
        if self.default_url:
            word = self.pos_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        out_text = in_text + ' ' + link
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
            elif task_config['task_name'] == 'sentiment':
                y_out = smooth_label(y, factor=0.5)
            else:
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class AddNegativeLink(AddSentimentLink):
    def __init__(self, return_metadata=False):
        super().__init__(url=None, sentiment='negative', return_metadata=False)
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]

    def __call__(self, in_text):
        if self.default_url:
            word = self.neg_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        out_text = in_text + ' ' + link
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
            elif task_config['task_name'] == 'sentiment':
                y_out = smooth_label(y, factor=0.5)
            else:
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out