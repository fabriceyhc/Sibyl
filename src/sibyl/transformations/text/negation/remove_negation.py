from ..abstract_transformation import *
import collections
import pattern
import spacy
import en_core_web_sm

class RemoveNegation(AbstractTransformation):
    """
    Defines a transformation that removes a negation
    if found within the input. Otherwise, return the 
    original string unchanged. 
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
        """Removes negation from a string by first 
        converting it to spacy.token.Doc. 

        Parameters
        ----------
        string : str
            Input to have negation removed

        Returns
        -------
        ret : str
            Output with *all* negations removed

        """
        # This removes all negations in the doc. I should maybe add an option to remove just some.
        doc = self.nlp(string)
        doc_len = len(doc)
        notzs = [i for i, z in enumerate(doc) if z.lemma_ == 'not' or z.dep_ == 'neg']
        new = []
        for notz in notzs:
            before = doc[notz - 1] if notz != 0 else None
            after = doc[notz + 1] if len(doc) > notz + 1 else None
            if (after and after.pos_ == 'PUNCT') or (before and before.text in ['or']):
                continue
            new.append(notz)
        notzs = new
        if not notzs:
            meta = {'change': False}
            if self.metadata: return string, meta
            return string
        ret = ''
        start = 0
        for i, notz in enumerate(notzs):
            id_start = notz
            to_add = ' '
            id_end = notz + 1
            before = doc[notz - 1] if notz != 0 else None
            after = doc[notz + 1] if len(doc) > notz + 1 else None
            if before and before.lemma_ in ['will', 'can', 'do']:
                id_start = notz - 1
                tenses = collections.Counter([x[0] for x in pattern.en.tenses(before.text)]).most_common(1)
                tense = tenses[0][0] if len(tenses) else 'present'
                p = pattern.en.tenses(before.text)
                params = [tense, 3]
                if p:
                    tmp = [x for x in p if x[0] == tense]
                    if tmp:
                        params = list(tmp[0])
                    else:
                        params = list(p[0])
                to_add = ' '+ pattern.en.conjugate(before.lemma_, *params) + ' '
            if before and after and before.lemma_ == 'do' and after.pos_ == 'VERB':
                id_start = notz - 1
                tenses = collections.Counter([x[0] for x in pattern.en.tenses(before.text)]).most_common(1)
                tense = tenses[0][0] if len(tenses) else 'present'
                p = pattern.en.tenses(before.text)
                params = [tense, 3]
                if p:
                    tmp = [x for x in p if x[0] == tense]
                    if tmp:
                        params = list(tmp[0])
                    else:
                        params = list(p[0])
                to_add = ' '+ pattern.en.conjugate(after.text, *params) + ' '
                id_end = notz + 2
            # print(doc)
            # print(start)
            # print(id_start)
            # print(to_add)
            if start == id_start or start >= doc_len or id_start >= doc_len:
                continue
            else:
                ret += doc[start:id_start].text + to_add
            start = id_end
        if id_end < doc_len:
            ret += doc[id_end:].text
        assert type(ret) == str
        meta = {'change': string!=ret}
        if self.metadata: return ret, meta
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
            soften = label_type == 'soft'
            y_ = invert_label(y, soften=soften)
        if self.metadata: return X_[0], y_, X_[1]
        return X_, y_