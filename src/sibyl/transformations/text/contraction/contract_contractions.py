from ..abstract_transformation import *
import re

class ContractContractions(AbstractTransformation):
    """
    Contracts all known contractions in a string input or 
    returns the original string if none are found. 
    """

    def __init__(self, task=None, return_metadata=False):
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
        a = {
            'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
            'could not': "couldn't", 'did not': "didn't", 'does not':
            "doesn't", 'do not': "don't", 'had not': "hadn't", 'has not':
            "hasn't", 'have not': "haven't", 'he is': "he's", 'how did':
            "how'd", 'how is': "how's", 'I would': "I'd", 'I will': "I'll",
            'I am': "I'm", 'i would': "i'd", 'i will': "i'll", 'i am': "i'm",
            'it would': "it'd", 'it will': "it'll", 'it is': "it's",
            'might not': "mightn't", 'must not': "mustn't", 'need not': "needn't",
            'ought not': "oughtn't", 'shall not': "shan't", 'she would': "she'd",
            'she will': "she'll", 'she is': "she's", 'should not': "shouldn't",
            'that would': "that'd", 'that is': "that's", 'there would':
            "there'd", 'there is': "there's", 'they would': "they'd",
            'they will': "they'll", 'they are': "they're", 'was not': "wasn't",
            'we would': "we'd", 'we will': "we'll", 'we are': "we're", 'were not':
            "weren't", 'what are': "what're", 'what is': "what's", 'when is':
            "when's", 'where did': "where'd", 'where is': "where's",
            'who will': "who'll", 'who is': "who's", 'who have': "who've", 'why is':
            "why's", 'will not': "won't", 'would not': "wouldn't", 'you would':
            "you'd", 'you will': "you'll", 'you are': "you're",
        }
        b = {"ain't": "isn't", "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "could've": 'could have', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'll": 'I will', "I'm": 'I am', "I've": 'I have', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it will', "it's": 'it is', "ma'am": 'madam', "might've": 'might have', "mightn't": 'might not', "must've": 'must have', "mustn't": 'must not', "needn't": 'need not', "oughtn't": 'ought not', "shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "that'd": 'that would', "that's": 'that is', "there'd": 'there would', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "wasn't": 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what're": 'what are', "what's": 'what is', "when's": 'when is', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'll": 'who will', "who's": 'who is', "who've": 'who have', "why's": 'why is', "won't": 'will not', "would've": 'would have', "wouldn't": 'would not', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will', "you're": 'you are', "you've": 'you have'}
        reverse_b = {v: k for k, v in b.items()}
        self.reverse_contraction_map = {**a, **reverse_b}
        self.return_metadata = return_metadata
        
    def __call__(self, in_text):
        """Applies contractions to strings (if any)

        Parameters
        ----------
        in_text : str or list[str]
            String or list of strings

        Returns
        -------
        out_text : str or list[str]
            String or list of strings with contractions (if any)
        """
        reverse_contraction_pattern = re.compile(r'\b({})\b '.format(
            '|'.join(self.reverse_contraction_map.keys())), flags=re.IGNORECASE|re.DOTALL)
        def cont(possible):
            match = possible.group(1)
            first_char = match[0]
            expanded_contraction = self.reverse_contraction_map.get(match, 
                self.reverse_contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:] + ' '
            return expanded_contraction

        if isinstance(in_text, str):
            in_text = [in_text]

        out_text = []
        for t in in_text:
            out_text.append(reverse_contraction_pattern.sub(cont, t))

        metadata = {'change': in_text != out_text}
        out_text = out_text[0] if len(out_text) == 1 else out_text
        if self.return_metadata: 
            return out_text, metadata
        return out_text

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.task_config = [
            {
                'task_name' : 'sentiment',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'topic',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'grammaticality',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'similarity',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'entailment',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
            {
                'task_name' : 'qa',
                'tran_type' : 'INV',
                'label_type' : 'hard'
            },
        ]
        df = self._get_tran_types(self.task_config, task_name, tran_type, label_type)
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