from ..abstract_batch_transformation import AbstractBatchTransformation
from ...utils import one_hot_encode
import numpy as np
import pandas as pd
import itertools
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextMix(AbstractBatchTransformation):
    """
    Concatenates two texts together and interpolates 
    the labels
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, batch, target_pairs=[], target_prob=0, num_classes=2):
        """
        Parameters
        ----------
        batch : tuple(np.array, np.array)
            The input data and the targets, which will
            be one-hot-encoded if they aren't already
        target_pairs : list[tuple(int, int)]
            A list of tuple pairs that represent a mapping
            between a source_class and target_class. The 
            target_class gets cut into the source_class. 
            For all unspecified class pairings, a random 
            pairing will be made
        target_prob : float [0,1]
            The probability of applying the targeted cutmix
            for a particular pairing
        num_classes : int
            The number of classes in the dataset

        Returns
        ----------
        ret : tuple(np.array, np.array)
            : tuple(np.array, np.array, dict)
            The output batch of transformed (X, y)
            pairs. y --> one-hot-encoded, weighted
            by the amount of text drawn from each
            of the inputs. May return metadata dict
            if requested.
        """
        
        # unpack batch
        data, targets = batch
        batch_size = len(data)

        # convert to numpy if not already
        if type(data) == list:
            data = [x.encode('utf-8') for x in data]
            data = np.array(data, dtype=np.string_)
        if type(targets) == list:
            if type(targets[0]) == np.ndarray:
                targets = np.stack(targets)
            else:
                targets = np.array(targets)

        # track indices for targeted cutmix to exclude later
        idx = [x for x in range(batch_size)]
        ex_idx = []

        new_data = []
        new_targets = []

        # transform targeted pairings
        for pair in target_pairs:
            
            # skip targeted transformation target_prob percent of the time
            use_targets = np.random.uniform() < target_prob
            if not use_targets:
                continue
                
            # unpack source and target pairs
            source_class, target_class = pair

            # find indices of both source and target 
            s_idx = find_value_idx(targets, source_class)
            t_idx = find_value_idx(targets, target_class)
            
            # if none of the source or target classes are in this batch, skip it
            if len(s_idx) == 0 or len(t_idx) == 0:
                continue
                
            # enforce source==target array size via sampling
            tt_idx = np.random.choice(np.arange(len(t_idx)), size=len(s_idx), replace=True)
            t_idx = t_idx[tt_idx]
            
            # create concatenated data
            textmix = concat_text(data[s_idx], data[t_idx])
            ohe_targets = concat_labels(data[s_idx], 
                                        data[t_idx], 
                                        targets[s_idx],
                                        targets[t_idx],
                                        num_classes)
            
            new_data.append(textmix)
            new_targets.append(ohe_targets)
            ex_idx.append(s_idx.tolist())
            
        ex_idx = list(itertools.chain(*ex_idx))
        s_idx = [i for i in idx if i not in ex_idx]
        t_idx = np.random.choice(np.arange(len(s_idx)), size=len(s_idx), replace=True)

        if s_idx:
            textmix = concat_text(data[s_idx], data[t_idx])
            ohe_targets = concat_labels(data[s_idx], 
                                        data[t_idx], 
                                        targets[s_idx],
                                        targets[t_idx],
                                        num_classes)

            new_data.append(textmix)
            new_targets.append(ohe_targets)
            
        new_data = np.concatenate(new_data)
        new_data = [x.decode() if type(x) == bytes else str(x) for x in new_data]
        new_targets = np.concatenate(new_targets).tolist()

        ret = (new_data, new_targets)

        # metadata
        if self.metadata: 
            meta = {'change': True}
            return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'SIB'],
            'label_type': ['soft', 'soft']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df

class SentMix(AbstractBatchTransformation):
    """
    Concatenates two texts together and then mixes the
    sentences in the new string. Interpolates the labels
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, batch, target_pairs=[], target_prob=0, num_classes=2):
        """
        Parameters
        ----------
        batch : tuple(np.array, np.array)
            The input data and the targets, which will
            be one-hot-encoded if they aren't already
        target_pairs : list[tuple(int, int)]
            A list of tuple pairs that represent a mapping
            between a source_class and target_class. The 
            target_class gets cut into the source_class. 
            For all unspecified class pairings, a random 
            pairing will be made
        target_prob : float [0,1]
            The probability of applying the targeted cutmix
            for a particular pairing
        num_classes : int
            The number of classes in the dataset

        Returns
        ----------
        ret : tuple(np.array, np.array)
            : tuple(np.array, np.array, dict)
            The output batch of transformed (X, y)
            pairs. y --> one-hot-encoded, weighted
            by the amount of text drawn from each
            of the inputs. May return metadata dict
            if requested.
        """

        # TextMix transformation
        new_data, new_targets = TextMix()(batch, target_pairs, target_prob, num_classes)

        # mix sentences
        sent_shuffle_ = np.vectorize(sent_shuffle)
        new_data = np.apply_along_axis(sent_shuffle_, 0, new_data).tolist()

        ret = (new_data, new_targets)

        # metadata
        if self.metadata: 
            meta = {'change': True}
            return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'SIB'],
            'label_type': ['soft', 'soft']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df

class WordMix(AbstractBatchTransformation):
    """
    Concatenates two texts together and then mixes the
    words in the new string. Interpolates the labels
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, batch, target_pairs=[], target_prob=0, num_classes=2):
        """
        Parameters
        ----------
        batch : tuple(np.array, np.array)
            The input data and the targets, which will
            be one-hot-encoded if they aren't already
        target_pairs : list[tuple(int, int)]
            A list of tuple pairs that represent a mapping
            between a source_class and target_class. The 
            target_class gets cut into the source_class. 
            For all unspecified class pairings, a random 
            pairing will be made
        target_prob : float [0,1]
            The probability of applying the targeted cutmix
            for a particular pairing
        num_classes : int
            The number of classes in the dataset

        Returns
        ----------
        ret : tuple(np.array, np.array)
            : tuple(np.array, np.array, dict)
            The output batch of transformed (X, y)
            pairs. y --> one-hot-encoded, weighted
            by the amount of text drawn from each
            of the inputs. May return metadata dict
            if requested.
        """

        # TextMix transformation
        new_data, new_targets = TextMix()(batch, target_pairs, target_prob, num_classes)

        # mix words
        word_shuffle_ = np.vectorize(word_shuffle)
        new_data = np.apply_along_axis(word_shuffle_, 0, new_data).tolist()

        ret = (new_data, new_targets)

        # metadata
        if self.metadata: 
            meta = {'change': True}
            return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'SIB'],
            'label_type': ['soft', 'soft']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df

# Helper functions
def concat_labels(source_text, 
                  target_text, 
                  source_labels, 
                  target_labels, 
                  num_classes):
    
    # create soft target labels 
    source_ohe = one_hot_encode(source_labels, num_classes)
    target_ohe = one_hot_encode(target_labels, num_classes)

    if len(source_ohe.shape) == 1:
        source_ohe = np.expand_dims(source_ohe, 0)
    if len(target_ohe.shape) == 1:
        target_ohe = np.expand_dims(target_ohe, 0)
    
    if source_labels.shape[-1] == 1:
        source_cls = source_labels
        target_cls = target_labels
    else:
        source_cls = np.argmax(source_ohe, axis=1)
        target_cls = np.argmax(target_ohe, axis=1)
        
    # calculate length of each data and use that
    # to determine the lambda weight assigned to
    # the index for the target
    len_data_source = np.char.str_len(source_text)
    len_data_target = np.char.str_len(target_text)
    lam = len_data_source / (len_data_source + len_data_target)   

    idx_ = np.arange(len(source_ohe))

    # print(lam.shape, lam)
    # print(idx_.shape, idx_)
    # print(source_ohe.shape, source_ohe)
    # print(target_ohe.shape, target_ohe)
    # print(source_cls.shape, source_cls)
    # print(target_cls.shape, target_cls)
    
    # NOTE: https://stackoverflow.com/questions/38673531/numpy-cannot-cast-ufunc-multiply-output-from-dtype
    source_ohe[idx_, source_cls] = (source_ohe[idx_, source_cls] * lam)
    target_ohe[idx_, target_cls] = (target_ohe[idx_, target_cls] * 1-lam)
    
    ohe_targets = source_ohe + target_ohe
    
    return ohe_targets

def concat_text(np_char1, np_char2):
    np_char1 = np_char1.astype(np.string_)
    np_char2 = np_char2.astype(np.string_)
    sep = np.full_like(np_char1, " ", dtype=np.string_)
    ret = np.char.add(np_char1, sep)
    ret = np.char.add(ret, np_char2)
    return ret

def sent_shuffle(string):
    X = nltk.tokenize.sent_tokenize(str(string))
    np.random.shuffle(X)
    return ' '.join(X)

def word_shuffle(string):
    X = str(string).split(" ")
    np.random.shuffle(X)
    return ' '.join(X)

def find_value_idx(array, value):
    return np.asarray(array == value).nonzero()[0]

def find_other_idx(array, value):
    return np.asarray(array != value).nonzero()[0]