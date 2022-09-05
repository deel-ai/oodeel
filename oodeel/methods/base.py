from typing import Type, Union, Iterable, Callable
import tensorflow as tf
from abc import ABC, abstractmethod
from ..utils.tf_feature_extractor import FeatureExtractor

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
    def __init__(
        self,
        output_layers=[-1], 
        output_activations=["same"], 
        flatten=True, 
        batch_size=256,
        threshold=None
    ):

        self.batch_size = batch_size
        self.threshold = threshold
        self.feature_extractor = None 
        self.output_layers = output_layers
        self.output_activations = output_activations
        self.flatten = flatten


    @abstractmethod
    def _score_tensor(self, inputs):
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


    def load(self, model, fit_dataset=None):
        """
        Prepare oodmodel for scoring:
        * Load the feature extractor
        * Calibrates the model on ID data "id_dataset".

        Parameters
        ----------
        id_dataset : np.array
            ID dataset that the method has to be calibrated on

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.feature_extractor = self._load_feature_extractor(model)
        if fit_dataset is not None:
            self._fit_to_dataset(fit_dataset)


    def _load_feature_extractor(self, model):
        """
        Loads feature extractor

        Args:
            model : tf.keras model 
                keras models saved as pb files e.g. with model.save()
        """
        feature_extractor = FeatureExtractor(model, 
                                                    output_layers=self.output_layers, 
                                                    output_activations=self.output_activations, 
                                                    flatten=self.flatten,
                                                    batch_size=self.batch_size)
        return feature_extractor

    def _fit_to_dataset(self, fit_dataset):
        """
        Loads feature extractor

        Args:
            model : tf.keras model 
                keras models saved as pb files e.g. with model.save()
        """
        raise NotImplementedError()

    def calibrate_threshold(self, fit_dataset, scores):
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
        if type(inputs) is not list:
            scores = self._score_tensor(inputs)
            return scores
        else:
            scores_list = []
            for input in inputs:
                scores = self._score_tensor(input)
                scores_list.append(scores)
            return scores_list

    def isood(self, inputs):
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

        if (self.scores is None) or (inputs is not None):
            self.score(inputs)
        OODness = tf.map_fn(lambda x: 0 if x < self.threshold else 1, self.scores)

        return OODness

    def __call__(self, inputs):
        """
        Convenience wrapper for isood once the threshold is set
        """
        return self.isood(inputs)
            






    


