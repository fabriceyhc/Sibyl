from ..abstract_transformation import *
import collections
import pattern
from pattern import en
import spacy
import en_core_web_sm

class AddNegation(AbstractTransformation):
    """
    Defines a transformation that negates a string.
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
        self.nlp = en_core_web_sm.load()
        self.metadata = meta
    
    def __call__(self, string):
        """Adds negation to a string by first 
        converting it to spacy.token.Doc. 

        Parameters
        ----------
        string : str
            Input to have negation added

        Returns
        -------
        ret : str
            Output with *all* negations added

        """
        ans = self.wrapper(string)
        if ans is None: ans = string
        meta = {'change': string!=ans}
        if self.metadata: return ans, meta
        return ans

    def wrapper(self, string):
        doc = self.nlp(string)
        doc_len = len(doc)
        for sentence in doc.sents:
            if len(sentence) < 3:
                continue
            root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
            root = doc[root_id]
            if '?' in sentence.text and sentence[0].text.lower() == 'how':
                continue
            if root.lemma_.lower() in ['thank', 'use']:
                continue
            if root.pos_ not in ['VERB', 'AUX']:
                continue
            neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
            if neg:
                continue
            if root.lemma_ == 'be':
                if '?' in sentence.text:
                    continue
                if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
                    if root_id + 1 >= doc_len:
                        ret = string
                    else:
                        ret = doc[:root_id + 1].text + ' not ' + doc[root_id + 1:].text
                    assert type(ret) == str
                    return ret
                else:
                    if root_id == 0:
                        ret = doc[root_id].text + ' not ' + doc[root_id:].text
                    else:    
                        ret = doc[:root_id].text + ' not ' + doc[root_id:].text
                    assert type(ret) == str
                    return ret
            else:
                aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
                if aux:
                    aux = aux[0]
                    if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
                        lemma = doc[aux.i].lemma_.lower()
                        if lemma == 'will':
                            fixed = 'won\'t'
                        elif lemma == 'have' and doc[aux.i].text in ['\'ve', '\'d']:
                            fixed = 'haven\'t' if doc[aux.i].text == '\'ve' else 'hadn\'t'
                        elif lemma == 'would' and doc[aux.i].text in ['\'d']:
                            fixed = 'wouldn\'t'
                        else:
                            fixed = doc[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
                        fixed = ' %s ' % fixed
                        if aux.i == 0:
                            ret = doc[aux.i].text + fixed + doc[aux.i + 1:].text
                        else:
                            ret = doc[:aux.i].text + fixed + doc[aux.i + 1:].text
                        assert type(ret) == str
                        return ret
                    if root_id  == 0:
                        ret = doc[root_id].text + ' not ' + doc[root_id:].text
                    else:
                        ret = doc[:root_id].text + ' not ' + doc[root_id:].text
                    assert type(ret) == str
                    return ret
                else:
                    # TODO: does, do, etc. 
                    subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
                    p = pattern.en.tenses(root.text)
                    tenses = collections.Counter([x[0] for x in pattern.en.tenses(root.text)]).most_common(1)
                    tense = tenses[0][0] if len(tenses) else 'present'
                    params = [tense, 3]
                    if p:
                        tmp = [x for x in p if x[0] == tense]
                        if tmp:
                            params = list(tmp[0])
                        else:
                            params = list(p[0])
                    if root.tag_ not in ['VBG']:
                        do = pattern.en.conjugate('do', *params) + 'n\'t'
                        new_root = pattern.en.conjugate(root.text, tense='infinitive')
                    else:
                        do = 'not'
                        new_root = root.text
                    # print(doc)
                    # print(len(doc))
                    # print(root_id)
                    # print(do)
                    # print(new_root)
                    # print(doc[root_id + 1:].text)
                    if root_id == 0 or root_id + 1 >= doc_len:
                        ret = string
                    else:
                        ret = '%s %s %s %s' % (doc[:root_id].text, do, new_root, doc[root_id + 1:].text)
                    assert type(ret) == str
                    return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV'],
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