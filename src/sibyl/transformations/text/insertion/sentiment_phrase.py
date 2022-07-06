from ..abstract_transformation import *
from ..tasks import *
from ..data.phrases import POSITIVE_PHRASES, NEGATIVE_PHRASES
import numpy as np

class InsertSentimentPhrase(AbstractTransformation):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, sentiment='positive', return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        sentiment : str
            Determines whether the inserted phraase will 
            feature a positive or negative sentiment.
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        super().__init__() 
        self.sentiment = sentiment
        if self.sentiment.lower() not in ['positive', 'negative']:
            raise ValueError("Sentiment must be 'positive' or 'negative'.")
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
        """
        Appends a sentiment-laden phrase to a string.
        """
        if 'positive' in self.sentiment:
        	phrase = self.np_random.choice(POSITIVE_PHRASES)
        if 'negative' in self.sentiment:
        	phrase = self.np_random.choice(NEGATIVE_PHRASES)
        out_text = in_text + " " + phrase
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
            if self.task_config['task_name'] == 'similarity':
                # hard code for now... :(
                # 0 = dissimilar, 1 = similar
                if y == 0:
                    y_out = 0
                else:
                    y_out = smooth_label(y, factor=0.5)
            elif self.task_config['task_name'] == 'sentiment':
                if self.sentiment == 'positive':
                    y_out = smooth_label(y, factor=0.5)
                if self.sentiment == 'negative':
                    y_out = smooth_label(y, factor=0.5)
            else:
                soften = self.task_config['label_type'] == 'soft'
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class InsertPositivePhrase(InsertSentimentPhrase):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, return_metadata=False):
        super().__init__(sentiment = 'positive', return_metadata=False)
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
        phrase = self.np_random.choice(POSITIVE_PHRASES)
        out_text = in_text + " " + phrase
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
            if self.task_config['task_name'] == 'similarity':
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
            elif self.task_config['task_name'] == 'sentiment':
                y_out = smooth_label(y, factor=0.5)
            else:
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class InsertNegativePhrase(InsertSentimentPhrase):
    """
    Appends a sentiment-laden phrase to a string based on 
    a pre-defined list of phrases.  
    """

    def __init__(self, return_metadata=False):
        super().__init__(sentiment = 'negative', return_metadata=False)
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
        phrase = self.np_random.choice(NEGATIVE_PHRASES)
        out_text = in_text + " " + phrase
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
            if self.task_config['task_name'] == 'similarity':
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
            elif self.task_config['task_name'] == 'sentiment':
                y_out = smooth_label(y, factor=0.5)
            else:
                y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out