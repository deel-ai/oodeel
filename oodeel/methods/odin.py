import tensorflow as tf
from .base import OODModel
import numpy as np
import faiss
from ..types import *
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


class ODIN(OODModel):
    """
    "Out-of-Distribution Detection with Deep Nearest Neighbors"
    https://arxiv.org/abs/2204.06507
    Simplified version adapted to convnet as built in ./models/train/train_mnist.py

        Args:
            nearest: number of nearest neighbors to consider.
                Defaults to 1.
            output_layers_id: feature space on which to compute nearest neighbors.
                Defaults to [-2].
            output_activation: output activation to use.
                Defaults to None.
            flatten: Flatten the output features or not.
                Defaults to True.
            batch_size: batch_size used to compute the features space
                projection of input data.
                Defaults to 256.
    """

    def __init__(
        self,
        temperature: float = 1000,
        noise: float = 0.2,
        batch_size: int = 256,
    ):
        self.temperature = temperature
        softmax_temp = init_softmax_temp(temperature)
        super().__init__(output_activation=softmax_temp, batch_size=batch_size)
        self.noise = noise

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
        grads = self.feature_extractor.gradient_max(inputs)
        inputs = inputs[0] - self.noise * np.sign(-grads)
        pred = self.feature_extractor(inputs)[0]
        scores = -np.max(pred, axis=1)
        return scores


def init_softmax_temp(temperature: float = 1):
    """
    Adds softmax_temp to the list of keras activations

    Args:
        temperature: float
    """

    def softmax_temp(inputs):
        return tf.keras.activations.softmax(inputs / temperature)

    return Activation(softmax_temp)
