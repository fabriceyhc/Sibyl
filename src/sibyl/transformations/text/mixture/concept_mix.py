from ..abstract_batch_transformation import AbstractBatchTransformation
from ...utils import one_hot_encode
from ..tasks import *
from ..generative.concept2sentence import Concept2Sentence
import numpy as np
import pandas as pd
import itertools
import nltk

class ConceptMix(AbstractBatchTransformation):

    uses_dataset = True
    
    def __init__(self, 
                 return_metadata=False,
                 dataset=None, 
                 extract="token",
                 gen_beam_size=10,
                 text_min_length=10,
                 text_max_length=32,
                 device='cuda',
                 generation_type='disjoint',
                 task_config=None):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(tran_type='SIB'),
            Grammaticality(),
            # Similarity(input_idx=[1,0], tran_type='SIB'),
            # Similarity(input_idx=[0,1], tran_type='SIB'),
            # Similarity(input_idx=[1,1], tran_type='SIB'),
            # Entailment(input_idx=[1,0], tran_type='SIB'),
            # Entailment(input_idx=[0,1], tran_type='SIB'),
            # Entailment(input_idx=[1,1], tran_type='SIB'),
        ]
        self.c2s = Concept2Sentence(
                        dataset = dataset,
                        extract = extract,
                        gen_beam_size = gen_beam_size,
                        text_min_length = text_min_length,
                        text_max_length = text_max_length,
                        device = device)
        self.generation_type = generation_type
        if self.generation_type not in ['joint', 'disjoint']:
            raise ValueError("Must select a generation type from ['joint', 'disjoin']")
        
    def __call__(self, batch, target_pairs=[], target_prob=1, num_classes=2):
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
            The probability of applying the targeted mixing
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
            data = [str(x).encode('utf-8') for x in data]
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
            new_text, new_target = self.conmix(data[s_idx], 
                                               data[t_idx], 
                                               targets[s_idx],
                                               targets[t_idx],
                                               num_classes)
            
            new_data.append(new_text)
            new_targets.append(new_target)
            ex_idx.append(s_idx.tolist())
            
        ex_idx = list(itertools.chain(*ex_idx))
        s_idx = [i for i in idx if i not in ex_idx]
        t_idx = np.random.choice(np.arange(len(s_idx)), size=len(s_idx), replace=True)

        if s_idx:
            new_text, new_target = self.conmix(data[s_idx], 
                                               data[t_idx], 
                                               targets[s_idx],
                                               targets[t_idx],
                                               num_classes)

            new_data.append(new_text)
            new_targets.append(new_target)
            
        new_data = np.concatenate(new_data)
        new_data = [x.decode() if type(x) == bytes else str(x) for x in new_data]
        new_targets = np.concatenate(new_targets).tolist()

        out = (new_data, new_targets)

        # metadata
        if self.return_metadata: 
            metadata = {'change': True}
            return out, metadata
        return out

    def get_task_configs(self, task_name=None, tran_type=None, label_type=None):
        init_configs = [task() for task in self.task_configs]
        df = self._get_task_configs(init_configs, task_name, tran_type, label_type)
        return df

    def conmix(self, texts1, texts2, targets1, targets2, num_classes):

        # prep text
        texts1 = texts1.astype(np.string_)
        texts2 = texts2.astype(np.string_)
        # prep targets
        targets1_ohe = np.array([one_hot_encode(t, num_classes).astype(np.float32) for t in targets1])
        targets2_ohe = np.array([one_hot_encode(t, num_classes).astype(np.float32) for t in targets2])
        if len(targets1_ohe.shape) == 1:
            targets1_ohe = np.expand_dims(targets1_ohe, 0)
        if len(targets2_ohe.shape) == 1:
            targets2_ohe = np.expand_dims(targets2_ohe, 0)
        if targets1.shape[-1] == 1:
            targets1_cls = targets1
            targets2_cls = targets2
        else:
            targets1_cls = np.argmax(targets1_ohe, axis=1)
            targets2_cls = np.argmax(targets2_ohe, axis=1)

        # print('texts1', texts1)
        # print('texts2', texts2)
        # print('targets1', targets1)
        # print('targets2', targets2)
        # print('targets1_ohe', targets1_ohe)
        # print('targets2_ohe', targets2_ohe)
        # print('targets1_cls', targets1_cls)
        # print('targets2_cls', targets2_cls)

        # transform
        new_texts, new_targets = [], []
        for s1, s2, t1_ohe, t2_ohe, t1_cls, t2_cls in zip(texts1, texts2, 
                                                          targets1_ohe, targets2_ohe, 
                                                          targets1_cls, targets2_cls):

            if isinstance(s1, bytes):
                s1 = s1.decode()
            if isinstance(s2, bytes):
                s2 = s2.decode()

            # transform text
            concepts1 = self.c2s.extract_concepts(s1, t1_cls)
            concepts2 = self.c2s.extract_concepts(s2, t2_cls)
            combined_concepts = list(set(concepts1 + concepts2))

            if not combined_concepts:
                # if there aren't any concepts extracted
                # simply append the original sentence and target unchanged
                new_texts.append(s1)
                new_targets.append(t1_ohe)

            else:
                if self.generation_type == 'joint':

                    new_text = self.c2s.generate_text_from_concepts(combined_concepts)

                elif self.generation_type == 'disjoint':

                    if concepts1:
                        new_text1 = self.c2s.generate_text_from_concepts(concepts1)
                    else:
                        new_text1 = s1

                    if not concepts2:
                        new_text2 = self.c2s.generate_text_from_concepts(concepts2)
                    else:
                        new_text2 = s2

                    new_text = new_text1 + " " + new_text2

                # transform targets
                lam = len(concepts1) / len(combined_concepts)
                t1_ohe[t1_cls] *= lam
                t2_ohe[t2_cls] *= 1-lam    
                new_target = np.array(t1_ohe) + np.array(t2_ohe)

                new_texts.append(new_text)
                new_targets.append(new_target)

        return new_texts, new_targets

# Helper functions

def find_value_idx(array, value):
    return np.asarray(array == value).nonzero()[0]

def find_other_idx(array, value):
    return np.asarray(array != value).nonzero()[0]

# if __name__ == '__main__':
#     import json
#     tf = ConceptMix(dataset='sst2')
#     texts = ["I hate how long loading the models takes to select better keyphrases.",
#              "I really love this movie a lot!"]
#     targets = [0, 1]
#     batch = (texts, targets)
#     results = []
#     new_text, new_target = tf(batch, num_classes=2)
#     results.append({
#         "class": tf.name(),
#         "inputs": {"texts": texts, "targets": targets}, 
#         "outputs": {"new_text": new_text, "new_target": new_target}
#     })
#     json_file = {"type": tf.name(), "results": results}
#     print(json.dumps(json_file, indent=2))