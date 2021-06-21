from ..abstract_transformation import *
import pandas as pd
import os

class AddSentimentLink(AbstractTransformation):
    """
    Appends a given / constructed URL to a string input.
    Current implementation constructs a default URL that
    makes use of dictionary.com and is sensitive to changes
    in routing structure. 
    """

    def __init__(self, url=None, sentiment='positive', task=None,meta=False):
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
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
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
        self.metadata = meta
    
    def __call__(self, string):
        """
        Appends a given / constructed URL to a string input.
        Parameters
        ----------
        string : str
            Input string
        Returns
        -------
        ret
            String with URL appended
        """
        if self.default_url:
            if 'positive' in self.sentiment:
                word = self.pos_words.sample(1)['word'].iloc[0]
            if 'negative' in self.sentiment:
                word = self.neg_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        ret = string + ' ' + link
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV'],
            'label_type': ['soft', 'hard']
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
        if tran_type == 'SIB':
            if self.sentiment == 'positive':
                y_ = smooth_label(y, factor=0.5)
            if self.sentiment == 'negative':
                y_ = smooth_label(y, factor=0.5)
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class AddPositiveLink(AddSentimentLink):
    def __init__(self, task=None, meta=False):
        super().__init__(url=None, sentiment='positive', task=task, meta=meta)
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        if self.default_url:
            word = self.pos_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        ret = string + ' ' + link
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV'],
            'label_type': ['soft', 'hard']

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
        if tran_type == 'SIB':
            y_ = smooth_label(y, factor=0.5)
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class AddNegativeLink(AddSentimentLink):
    def __init__(self, task=None, meta=False):
        super().__init__(url=None, sentiment='negative', task=task, meta=meta)
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        if self.default_url:
            word = self.neg_words.sample(1)['word'].iloc[0]
            link = 'https://www.dictionary.com/browse/' + word
        else:
            link = self.url
        ret = string + ' ' + link
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV'],
            'label_type': ['soft', 'hard']
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
        if tran_type == 'SIB':
            y_ = smooth_label(y, factor=0.5)
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_