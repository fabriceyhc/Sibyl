from ..abstract_transformation import *
from urllib.request import urlopen
from bs4 import BeautifulSoup 
from bs4.element import Comment
import re

class ImportLinkText(AbstractTransformation):
    """
    Appends a given / constructed URL to a string input.
    Current implementation constructs a default URL that
    makes use of dictionary.com and is sensitive to changes
    in routing structure. 
    """

    def __init__(self, task=None,meta=False):
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
        # https://gist.github.com/uogbuji/705383#gistcomment-2250605
        self.URL_REGEX = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        self.metadata = meta
    
    def __call__(self, string):
        """
        Add extracted (visible)

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with visible text from the URL appended
        """
        def replace(match):
            url = match.group(0)
            return get_url_text(url)
        ret = self.URL_REGEX.sub(replace, string)
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