from ..abstract_transformation import *
from ..tasks import *

class WordDeletion(AbstractTransformation):
    """
    Deletes words from random indices in the string input
    """

    def __init__(self, p=0.25, task_name=None, return_metadata=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        p : float
            Randomly delete words from the sentence with probability p
        return_metadata : bool
            whether or not to return metadata, e.g. 
            whether a transform was successfully
            applied or not
        """
        super().__init__(task_name) 
        self.p = p
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
        """
        Parameters
        ----------
        in_text : str
            The input string

        Returns
        ----------
        out_text : str
            The output with random words deleted
        """
        words = in_text.split()
        # obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return in_text

        #randomly delete words with probability p
        new_words = []
        for word in words:
            r = self.np_random.uniform(0, 1)
            if r > self.p:
                new_words.append(word)

        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = self.np_random.integers(0, len(words)-1)
            return words[rand_int]

        out_text = ' '.join(new_words)
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
                else:
                    y_out = invert_label(y, soften=soften)
        
        if self.return_metadata: 
            return X_out, y_out, metadata
        return X_out, y_out