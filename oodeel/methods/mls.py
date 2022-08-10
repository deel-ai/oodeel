import tensorflow as tf
from .base import OODModel
import numpy as np

class MLS(OODModel):
    """
    Maximum Logit Scores method for OOD detection.
    "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?"
    https://arxiv.org/abs/2110.06207

    Parameters
    ----------
    model : tf.keras model 
        keras models saved as pb files e.g. with model.save()
    """
    def __init__(self, model):
        """
        Initializes the feature extractor 
        """
        super().__init__(model)
        feature_extractor = tf.keras.models.clone_model(model)
        feature_extractor.set_weights(model.get_weights())
        feature_extractor.layers[-1].activation = tf.keras.activations.linear
        self.feature_extractor = feature_extractor

    def score(self, inputs):
        """
        Computes an OOD score for input samples "inputs" based on 
        maximum logits value.

        Parameters
        ----------
        inputs : np.array
            input samples to score

        Returns
        -------
        np.array
            scores
        """
        pred = self.feature_extractor.predict(inputs)
        scores = np.max(pred, axis=1)
        self.scores = scores
        return scores

        