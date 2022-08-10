from typing import Type, Union, Iterable, Callable
import tensorflow as tf
from abc import ABC, abstractmethod

class OODModel(ABC):
    """
    Base Class for methods that assign a score to unseen samples.

    Parameters
    ----------
    model : tf.keras model 
        keras models saved as pb files e.g. with model.save()
    threshold : float, optional
            threshold to use for distinguishing between OOD and ID, by default None
    """
    def __init__(self, model, threshold=None):
        self.threshold = threshold
        self.model = model
        self.feature_extractor = None
        self.scores = None


    @abstractmethod
    def score(self, inputs):
        """
        Computes an OOD score for input samples "inputs"

        Parameters
        ----------
        inputs : np.array
            input samples to score

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError()

    def isood(self, threshold, inputs=None):
        """
        Returns whether the input samples "inputs" are OOD or not, given a threshold

        Parameters
        ----------
        threshold : float
            threshold to use for distinguishing between OOD and ID
        inputs : np.array, optional
            input samples to score if no scores are saved, by default None
   
        Returns
        -------
        np.array
            array filled with 0 for ID samples and 1 for OOD samples
        """
        self.threshold = threshold
        if (self.scores is None) or (inputs is not None):
            self.score(inputs)
        OODness = tf.map_fn(lambda x: 0 if x < self.threshold else 1, self.scores)

        return OODness

    def __call__(self, inputs):
        """
        Convenience wrapper for isood once the threshold is set
        """
        return self.isood(inputs)
            

class OODModelWithId(OODModel):
    """
    Base Class for methods that assign a score to unseen samples, based on 
    a comparison with some ID data.

    Parameters
    ----------
    model : tf.keras model 
        keras models saved as pb files e.g. with model.save()
    threshold : float, optional
            threshold to use for distinguishing between OOD and ID, by default None
    """
    def __init__(self, model, threshold=None):
        super().__init__(model, threshold)
        self.id_projected = None

    def project_id(self, inputs):
        """
        Computes the representation of input samples "inputs" in some feature 
        space constructed out of self.model.

        Parameters
        ----------
        inputs : np.array
            input samples to score if no scores are saved

        Returns
        -------
        np.array
            projected inputs
        """
        id_projected = self.feature_extractor(inputs)
        return id_projected

    @abstractmethod
    def fit(self, id_dataset):
        """
        Calibrates the model on ID data "id_dataset".

        Parameters
        ----------
        id_dataset : np.array
            ID dataset that the method has to be calibrated on

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError()




    


