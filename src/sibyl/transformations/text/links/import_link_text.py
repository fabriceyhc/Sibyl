from ..abstract_transformation import *
from ..tasks import *
from urllib.request import urlopen
from bs4 import BeautifulSoup 
from bs4.element import Comment
import re
import numpy as np

class ImportLinkText(AbstractTransformation):
    """
    Appends a given / constructed URL to a string input.
    Current implementation constructs a default URL that
    makes use of dictionary.com and is sensitive to changes
    in routing structure. 
    """

    def __init__(self, task_name=None, return_metadata=False):
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
        super().__init__(task_name) 
        # https://gist.github.com/uogbuji/705383#gistcomment-2250605
        self.URL_REGEX = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        self.return_metadata = return_metadata
        self.task_configs = [
            SentimentAnalysis(),
            TopicClassification(),
            Grammaticality(tran_type='SIB'),
            Similarity(input_idx=[1,0], tran_type='SIB'),
            Similarity(input_idx=[0,1], tran_type='SIB'),
            Similarity(input_idx=[1,1], tran_type='INV'),
            Entailment(input_idx=[1,0], tran_type='SIB'),
            Entailment(input_idx=[0,1], tran_type='SIB'),
            Entailment(input_idx=[1,1], tran_type='SIB'),
        ]
        self.task_config = self.match_task(task_name)
    
    def __call__(self, in_text):
        def replace(match):
            url = match.group(0)
            return get_url_text(url)
        out_text = self.URL_REGEX.sub(replace, in_text)
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
                elif self.task_config['task_name'] == 'entailment':
                    # hard coded for now... :(
                    # 0 = entailed, 1 = neutral, 2 = contradiction
                    if isinstance(y, int):
                        if y in [0, 2]:
                            y_out = 1
                        else: 
                            y_out = y
                    else:
                        if np.argmax(y) in [0, 2]:
                            y_out = 1
                        else:
                            y_out = y
                else:
                    y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_url_text(url):
    try:
        html = urlopen(url).read()
    except:
        return url
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)