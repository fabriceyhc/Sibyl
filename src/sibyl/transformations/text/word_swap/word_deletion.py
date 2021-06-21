from ..abstract_transformation import *
import random

class WordDeletion(AbstractTransformation):
    """
    Deletes words from random indices in the string input
    """

    def __init__(self, p=0.25, task=None,meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        p : float
            Randomly delete words from the sentence with probability p
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.p = p
        self.task = task
        self.metadata = meta
    
    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The input string

        Returns
        ----------
        ret : str
            The output with random words deleted
        """
        words = string.split()
        # obviously, if there's only one word, don't delete it
        if len(words) == 1:
            meta = {'change': string!=words}
            if self.metadata: return words, meta
            return words

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            meta = {'change': string!=words[rand_int]}
            if self.metadata: return words[rand_int], meta
            return words[rand_int]

        ret = ' '.join(new_words)
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