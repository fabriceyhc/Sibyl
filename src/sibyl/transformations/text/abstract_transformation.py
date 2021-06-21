from abc import ABC, abstractmethod
from ..utils import *
import pandas as pd
 
class AbstractTransformation(ABC):
    """
    An abstract class for transformations to be applied 
    to input data. 
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        """
        pass
    
    @abstractmethod
    def __call__(self, string):
        """
        Apply the transformation to a string input

        Parameters
        ----------
        string : str
            the input string
        """
        pass

    @abstractmethod
    def transform_Xy(self, X, y):
        """
        Apply the transformation to a string input 
        and an int target label

        Parameters
        ----------
        X : str
            the input string
        y : int
            the target label

        Returns
        ----------
        X_ : str
            the transformed string
        y_ : int
            if SIB ==> transformed target label
            if INV ==> the original target label
        """
        pass

    @abstractmethod
    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        """
        See self._get_tran_types()
        """
        pass


    def _get_tran_types(self, tran_types, task_name=None, tran_type=None, label_type=None):
        """
        Defines the task and type of transformation (SIB or INV) 
        to determine the effect on the expected behavior (whether 
        to change the label if SIB, or leave the label alone if INV). 

        Parameters
        ----------
        task_name : str
            Filters the results for the requested task.
        tran_type : str
            Filters the results for the requested trans type,
            which is either 'INV' or 'SIB'.
        label_type : str
            Filters the results for the requested label type,
            which is either 'hard' or 'soft'.

        Returns
        -------
        df : pandas.DataFrame
            A pandas DataFrame containing:
                - task_name : str
                    short description of the task
                - tran_type : str
                    INV == invariant ==> output behavior does 
                    not change
                    SIB == sibylvariant ==> output behavior 
                    changes in some way
                - label_type : str
                    whether to use soft or hard labels
        """
        df = pd.DataFrame.from_dict(tran_types)
        if task_name is not None:
            task_names = set(df.task_name.tolist())
            if task_name not in task_names:
                raise ValueError('The selected task must be one of the following: {}'.format(', '.join(task_names)))
            df = df[df['task_name'] == task_name]
        if tran_type is not None:
            tran_types = set(df.tran_type.tolist())
            if tran_type not in tran_types:
                raise ValueError('The selected tran type must be one of the following: {}'.format(', '.join(tran_types)))
            df = df[df['tran_type'] == tran_type]
        if label_type is not None:
            label_types = set(df.label_type.tolist())
            if label_type not in label_types:
                raise ValueError('The selected label type must be one of the following: {}'.format(', '.join(tran_types)))
            df = df[df['label_type'] == label_type]
        return df