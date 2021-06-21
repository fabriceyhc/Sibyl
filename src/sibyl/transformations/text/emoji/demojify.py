from ..abstract_transformation import *
from emoji_translate import Translator, emoji_lis

class Demojify(AbstractTransformation):
    def __init__(self, exact_match_only=False, randomize=True, task=None,meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        
        Parameters
        ----------
        exact_match_only : boolean
            Determines whether we find exact matches for
            the emoji's name for replacement. If false,
            approximate matching is used. 
        randomize : boolean
            If true, randomizes approximate matches.
            If false, always picks the first match.
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.exact_match_only = exact_match_only
        self.randomize = randomize
        self.emo = Translator(self.exact_match_only, self.randomize)
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The string input

        Returns
        ----------
        ret : str
            The output with all emojis removed.
        """
        ret = self.emo.demojify(string)
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

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

class RemoveEmoji(Demojify):
    def __init__(self, polarity=[-1, 1], task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        
        Parameters
        ----------
        polarity : list
            Emoji sentiment is measured in polarity 
            between -1 and +1. This param allows you
            to pick the sentiment range you want.
            - positivev ==> [0.05, 1]    
            - negative ==> [-1, -0.05] 
            - neutral ==> [-0.05, 0.05]
        """
        super().__init__(self, meta=meta) 
        self.polarity = polarity
        if self.polarity[0] <= -0.05:
            self.sentiment = 'negative'
        elif self.polarity[0] >= 0.05:
            self.sentiment = 'positive'
        else:
            self.sentiment = 'neutral'
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The string input

        Returns
        ----------
        ret : str
            The output with emojis in the target polarity
            range removed.
        """
        ret = self.remove_emoji_by_polarity(string, self.polarity)
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        pass

    def remove_emoji_by_polarity(self, string, p_rng=[-1,1]):
        for emoji_data in emoji_lis(string)[::-1]:
            i, emoji = emoji_data.values()
            _polarity = self.emo.get_df_by_emoji(emoji)['polarity'].iloc[0]
            if p_rng[0] <= _polarity <= p_rng[1]:
                string = string[:i] + '' + string[i + 1:].lstrip()
        return string.rstrip()

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
            if self.sentiment == 'neutral':
                y_ = y
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_

class RemovePositiveEmoji(RemoveEmoji):
    def __init__(self, polarity=[0.05, 1], task=None, meta=False):
        super().__init__(polarity=polarity, meta=meta) 
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        ret = self.remove_emoji_by_polarity(string, self.polarity)
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

class RemoveNegativeEmoji(RemoveEmoji):
    def __init__(self, polarity=[-1, -0.05], task=None, meta=False):
        super().__init__(polarity=polarity, meta=meta)
        self.task = task
        self.metadata = meta 

    def __call__(self, string):
        ret = self.remove_emoji_by_polarity(string, self.polarity)
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

class RemoveNeutralEmoji(RemoveEmoji):
    def __init__(self, polarity=[-0.05, 0.05], task=None, meta=False):
        super().__init__(polarity=polarity, meta=meta) 
        self.task = task
        self.metadata = meta

    def __call__(self, string):
        ret = self.remove_emoji_by_polarity(string, self.polarity)
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
        return ret

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
        if tran_type == 'SIB':
            y_ = y
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_