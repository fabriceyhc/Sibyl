from abc import ABC
 
class TaskConfig(ABC):
    """
    An abstract class defining the task configuration for transformations in the tool
    """

    def __init__(self, input_idx = [1], tran_type = 'INV', label_type = 'hard', **kwargs):
        """
        Initializes the task config
        """
        self.input_idx = input_idx
        self.tran_type = tran_type
        self.label_type = label_type
        if self.tran_type == 'INV':
            self.label_type = 'hard'
    
    def __call__(self):
        """
        Returns the task config as a dict
        """
        return {key : value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}

class SentimentAnalysis(TaskConfig):
    def __init__(self, input_idx = [1], tran_type = 'INV', label_type = 'hard'):
        super().__init__(input_idx=input_idx, tran_type=tran_type, label_type=label_type)
        self.task_name = "sentiment"


class TopicClassification(TaskConfig):
    def __init__(self, input_idx = [1], tran_type = 'INV', label_type = 'hard'):
        super().__init__(input_idx=input_idx, tran_type=tran_type, label_type=label_type)
        self.task_name = "topic"


class Grammaticality(TaskConfig):
    def __init__(self, input_idx = [1], tran_type = 'INV', label_type = 'hard'):
        super().__init__(input_idx=input_idx, tran_type=tran_type, label_type=label_type)
        self.task_name = "grammaticality"


class Similarity(TaskConfig):
    def __init__(self, input_idx = [1,1], tran_type = 'INV', label_type = 'hard'):
        super().__init__(input_idx=input_idx, tran_type=tran_type, label_type=label_type)
        self.task_name = "similarity"
    

class Entailment(TaskConfig):
    def __init__(self, input_idx = [1,1], tran_type = 'INV', label_type = 'hard'):
        super().__init__(input_idx=input_idx, tran_type=tran_type, label_type=label_type)
        self.task_name = "entailment"
  

class QuestionAndAnswer(TaskConfig):
    def __init__(self, input_idx = [1,1], tran_type = 'INV', label_type = 'hard'):
        super().__init__(input_idx=input_idx, tran_type=tran_type, label_type=label_type)
        self.task_name = "qa"