from ..abstract_transformation import *
from ..tasks import *
import numpy as np
import re
from pattern.en import wordnet
import nltk
from nltk.corpus import stopwords
import spacy
import en_core_web_sm

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ChangeSynse(AbstractTransformation):
    """
    Replaces a specified number of random words a string
    with synses from wordnet. Also supports part-of-speech (pos)
    tagging via spaCy to get more natural replacements. 
    """
    def __init__(self, synse='synonym', num_to_replace=-1, task_name=None, return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        synse : str
            The type of word replacement to make.
            Currently support 'synonym', 'antonym', 'hyponym', 
            and 'hypernym' 
        num_to_replace : int
            The number of words randomly selected from the input 
            to replace with synonyms (excludes stop words).
            If -1, then replace as many as possible.
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        super().__init__(task_name) 
        self.synse = synse
        self.synses = {
            'synonym' : all_possible_synonyms,
            'antonym' : all_possible_antonyms,
            'hyponym' : all_possible_hyponyms,
            'hypernym' : all_possible_hypernyms,
        }
        if synse not in self.synses:
            raise ValueError("Invalid synse argument. \n \
                Please select one of the following: 'synonym', 'antonym', 'hyponym', 'hypernym'")
        self.synse_fn = self.synses[self.synse]
        self.num_to_replace = np.Inf if num_to_replace == -1 else num_to_replace
        self.nlp = en_core_web_sm.load()
        # stopwords 
        useful_stopwords = ['few', 'more', 'most', 'against']
        self.stopwords = [w for w in stopwords.words('english') if w not in useful_stopwords]
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(tran_type='SIB'),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
        self.task_config = self.match_task(task_name)
    
    def __call__(self, in_text):
        """Replaces words with synses

        Parameters
        ----------
        in_text : str
            Input string

        Returns
        -------
        out_text : str
            Output string with synses replaced.
        """
        doc = self.nlp(in_text)
        new_words = in_text.split(' ').copy()
        random_word_list = list(set(
            [token.text for token in doc if strip_punct(token.text.lower()) not in self.stopwords]))
        self.np_random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            pos = [token.pos_ for token in doc if random_word in token.text][0]
            if pos in ['PROPN']:
                continue
            if pos not in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                pos = None
            options = self.synse_fn(strip_punct(random_word), pos)
            if len(options) >= 1:
                option = self.np_random.choice(list(options))
                new_words = [option if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= self.num_to_replace:
                break
        out_text = ' '.join(new_words)
        return out_text

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        pass

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
                if self.task_config['task_name'] == 'grammaticality':
                    # hard code for now... :(
                    # 0 = ungrammatical, 1 = grammatical
                    if isinstance(y, int):
                        if y == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                    else:
                        if np.argmax(y) == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                else:
                    y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class ChangeSynonym(ChangeSynse):
    def __init__(self, num_to_replace=-1, task_name=None, return_metadata=False):
        super().__init__(synse='synonym', num_to_replace=num_to_replace, task_name=task_name, return_metadata=False)
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(tran_type='SIB'),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
        self.task_config = self.match_task(task_name)

    def __call__(self, in_text):
        return super().__call__(in_text)

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
                if self.task_config['task_name'] == 'grammaticality':
                    # hard code for now... :(
                    # 0 = ungrammatical, 1 = grammatical
                    if isinstance(y, int):
                        if y == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                    else:
                        if np.argmax(y) == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                else:
                    y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class ChangeAntonym(ChangeSynse):
    def __init__(self, num_to_replace=-1, task_name=None, return_metadata=False):
        super().__init__(synse='antonym', num_to_replace=num_to_replace, task_name=task_name, return_metadata=False)
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(tran_type='SIB'),
            TopicClassification(),
            Grammaticality(tran_type='SIB'),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
        self.task_config = self.match_task(task_name)

    def __call__(self, in_text):
        return super().__call__(in_text)

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

class ChangeHyponym(ChangeSynse):
    def __init__(self, num_to_replace=-1, task_name=None, return_metadata=False):
        super().__init__(synse='hyponym', num_to_replace=num_to_replace, task_name=task_name, return_metadata=False)
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(tran_type='SIB'),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
        self.task_config = self.match_task(task_name)

    def __call__(self, in_text):
        return super().__call__(in_text)

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
                if self.task_config['task_name'] == 'grammaticality':
                    # hard code for now... :(
                    # 0 = ungrammatical, 1 = grammatical
                    if isinstance(y, int):
                        if y == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                    else:
                        if np.argmax(y) == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                elif self.task_config['task_name'] == 'similarity':
                    y_out = smooth_label(y, factor=0.25)
                else:
                    y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

class ChangeHypernym(ChangeSynse):
    def __init__(self, num_to_replace=-1, task_name=None, return_metadata=False):
        super().__init__(synse='hypernym', num_to_replace=num_to_replace, task_name=task_name, return_metadata=False)
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(tran_type='SIB'),
            Similarity(input_idx=[1,0], tran_type='INV'),
            Similarity(input_idx=[0,1], tran_type='INV'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='INV'),
            Entailment(input_idx=[0,1], tran_type='INV'),
            Entailment(input_idx=[1,1], tran_type='INV'),
        ]
        self.task_config = self.match_task(task_name)

    def __call__(self, in_text):
        return super().__call__(in_text)

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
                if self.task_config['task_name'] == 'grammaticality':
                    # hard code for now... :(
                    # 0 = ungrammatical, 1 = grammatical
                    if isinstance(y, int):
                        if y == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                    else:
                        if np.argmax(y) == 0:
                            y_out = y
                        else: 
                            y_out = invert_label(y, soften=soften)
                elif self.task_config['task_name'] == 'similarity':
                    y_out = smooth_label(y, factor=0.25)
                else:
                    y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

def all_synsets(word, pos=None):
    pos_map = {
        'NOUN': wordnet.NOUN,
        'VERB': wordnet.VERB,
        'ADJ': wordnet.ADJECTIVE,
        'ADV': wordnet.ADVERB
        }
    if pos is None:
        pos_list = [wordnet.VERB, wordnet.ADJECTIVE, wordnet.NOUN, wordnet.ADVERB]
    else:
        pos_list = [pos_map[pos]]
    out_text = []
    for pos in pos_list:
        out_text.extend(wordnet.synsets(word, pos=pos))
    return out_text

def clean_senses(synsets):
    return [x for x in set(synsets) if '_' not in x]

def all_possible_synonyms(word, pos=None):
    out_text = []
    for syn in all_synsets(word, pos=pos):
        # if syn.synonyms[0] != word:
        #     continue
        out_text.extend(syn.senses)
    return clean_senses(out_text)

def all_possible_antonyms(word, pos=None):
    out_text = []
    for syn in all_synsets(word, pos=pos):
        if not syn.antonym:
            continue
        for s in syn.antonym:
            out_text.extend(s.senses)            
    if not out_text:
        synos = all_possible_synonyms(word, pos)
        for syno in synos:
            for syn in all_synsets(syno, pos=pos):
                if not syn.antonym:
                    continue
                for s in syn.antonym:
                    out_text.extend(s.senses)
    return clean_senses(out_text)

def all_possible_hypernyms(word, pos=None, depth=3):
    out_text = []
    for syn in all_synsets(word, pos=pos):
        out_text.extend([y for x in syn.hypernyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(out_text)

def all_possible_hyponyms(word, pos=None, depth=3):
    out_text = []
    for syn in all_synsets(word, pos=pos):
        out_text.extend([y for x in syn.hyponyms(recursive=True, depth=depth) for y in x.senses])
    return clean_senses(out_text)

def strip_punct(word):
    puncts = re.compile(r'[^\w\s]')
    return puncts.sub('', word)