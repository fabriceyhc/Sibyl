from ..abstract_transformation import *
from ..tasks import *
from emoji_translate import Translator

class Emojify(AbstractTransformation):
    def __init__(self, exact_match_only=False, randomize=True, return_metadata=False):
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
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
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
        self.exact_match_only = exact_match_only
        self.randomize = randomize
        self.emo = Translator(self.exact_match_only, self.randomize)

    def __call__(self, in_text):
        out_text = self.emo.emojify(in_text)
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
            y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class AddEmoji(Emojify):
    def __init__(self, num=1, polarity=[-1, 1], return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        
        Parameters
        ----------
        num : int
            The number of emojis to append to the end
            of a given string
        polarity : list
            Emoji sentiment is measured in polarity 
            between -1 and +1. This param allows you
            to pick the sentiment range you want.
            - positivev ==> [0.05, 1]    
            - negative ==> [-1, -0.05] 
            - neutral ==> [-0.05, 0.05]
        """
        super().__init__(self, return_metadata=False) 
        self.num = num
        self.polarity = polarity
        if self.polarity[0] <= -0.05:
            self.sentiment = 'negative'
        elif self.polarity[0] >= 0.05:
            self.sentiment = 'positive'
        else:
            self.sentiment = 'neutral'
        
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

    def __call__(self, in_text):
        """
        Parameters
        ----------
        in_text : str
            The string input

        Returns
        ----------
        out_text : str
            The output with `num` emojis appended
        """
        out_text = in_text + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))
        return out_text

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        pass

    def sample_emoji_by_polarity(self, p_rng, num=1):
        emojis = self.emo.emojis
        return emojis[emojis['polarity'].apply(
            lambda x: p_rng[0] <= x <= p_rng[1])].sample(num)['char'].values.tolist()

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
            if self.sentiment == 'positive':
                y_out = smooth_label(y, factor=0.5)
            if self.sentiment == 'negative':
                y_out = smooth_label(y, factor=0.5)
            if self.sentiment == 'neutral':
                y_out = y
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class AddPositiveEmoji(AddEmoji):
    def __init__(self, num=1, polarity=[0.05, 1], return_metadata=False):
        super().__init__(self, return_metadata=False) 
        self.num = num
        self.polarity = polarity        
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]

    def __call__(self, in_text):
        """
        Parameters
        ----------
        in_text : str
            The string input

        Returns
        ----------
        out_text : str
            The output with `num` emojis appended
        """
        out_text = in_text + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))
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
            y_out = smooth_label(y, factor=0.5)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out


class AddNegativeEmoji(AddEmoji):
    def __init__(self, num=1, polarity=[-1, -0.05], return_metadata=False):
        super().__init__(self, return_metadata=False) 
        self.num = num
        self.polarity = polarity
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(),
            Grammaticality(),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]

    def __call__(self, in_text):
        """
        Parameters
        ----------
        in_text : str
            The string input

        Returns
        ----------
        out_text : str
            The output with `num` emojis appended
        """
        out_text = in_text + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))
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
            y_out = smooth_label(y, factor=0.5)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class AddNeutralEmoji(AddEmoji):
    def __init__(self, num=1, polarity=[-0.05, 0.05], return_metadata=False):
        super().__init__(self, return_metadata=False) 
        self.num = num
        self.polarity = polarity
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

    def __call__(self, in_text):
        """
        Parameters
        ----------
        in_text : str
            The string input

        Returns
        ----------
        out_text : str
            The output with `num` emojis appended
        """
        out_text = in_text + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))
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
            y_out = smooth_label(y, factor=0.5)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out