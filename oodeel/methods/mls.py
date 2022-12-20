import tensorflow as tf
from .base import OODModel
import numpy as np
from ..types import *


class MLS(OODModel):
    """
    Maximum Logit Scores method for OOD detection.
    "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?"
    https://arxiv.org/abs/2110.06207


    Args:
        output_activation: activation function for the last layer.
            Defaults to "linear".
        batch_size: batch_size used to compute the features space
            projection of input data.
            Defaults to 256.
    """

    def __init__(
        self,
        output_activation: str = "linear",
        batch_size: int = 256,
    ):
        super().__init__(output_activation=output_activation, batch_size=batch_size)

    def _score_tensor(
        self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"

        pred = self.feature_extractor(inputs)[0]
        scores = - np.max(pred, axis=1)
        return scores
