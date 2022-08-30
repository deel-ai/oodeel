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
    def __init__(
        self, 
        model, 
        output_activations=["linear"], 
        batch_size=256, 
        threshold=None
    ):
        """
        Initializes the feature extractor 
        """
        super().__init__(model=model, 
                         output_activations=output_activations, 
                         batch_size=batch_size,
                         threshold=threshold)

    def score_tensor(self, inputs):
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
        pred = self.feature_extractor(inputs)
        scores = np.max(pred, axis=1)
        self.scores = -scores
        return self.scores

        