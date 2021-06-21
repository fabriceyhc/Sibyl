from ..abstract_transformation import *
from ..data.phrases import POSITIVE_PHRASES, NEGATIVE_PHRASES
from random import sample 

class InsertSentimentPhrase(AbstractTransformation):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, sentiment='positive', task=None,meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        sentiment : str
            Determines whether the inserted phraase will 
            feature a positive or negative sentiment.
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.sentiment = sentiment
        if self.sentiment.lower() not in ['positive', 'negative']:
            raise ValueError("Sentiment must be 'positive' or 'negative'.")
        self.task = task
        self.metadata = meta
    
    def __call__(self, string):
        """
        Appends a sentiment-laden phrase to a string.

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with sentiment phrase appended
        """
        if 'positive' in self.sentiment:
        	phrase = sample(POSITIVE_PHRASES,1)[0]
        if 'negative' in self.sentiment:
        	phrase = sample(NEGATIVE_PHRASES,1)[0]
        ret = string + " " + phrase
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

class InsertPositivePhrase(InsertSentimentPhrase):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, task=None, meta=False):
        super().__init__(sentiment = 'positive', task=task, meta=meta)
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        phrase = sample(POSITIVE_PHRASES,1)[0]
        ret = string + " " + phrase
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

class InsertNegativePhrase(InsertSentimentPhrase):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, task=None, meta=False):
        super().__init__(sentiment = 'negative', task=task, meta=meta)
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        phrase = sample(NEGATIVE_PHRASES,1)[0]
        ret = string + " " + phrase
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