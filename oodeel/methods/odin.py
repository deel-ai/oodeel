# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import faiss
import numpy as np
import tensorflow as tf
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

from ..types import *
from ..utils.tf_operations import batch_tensor
from ..utils.tf_operations import find_layer
from ..utils.tf_operations import gradient
from ..utils.tf_operations import gradient_single
from ..utils.tools import dataset_get_columns
from ..utils.tools import dataset_nb_columns
from ..utils.tools import dataset_nb_labels
from .base import OODModel


class ODIN(OODModel):
    """
    "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks"
    http://arxiv.org/abs/1706.02690

    Parameters
    ----------
    temperature : float, optional
        Temperature parameter, by default 1000
    noise : float, optional
        Perturbation noise, by default 0.014
    batch_size : int, optional
        Batch size for score and perturbation computations, by default 256
    """

    def __init__(
        self,
        temperature: float = 1000,
        noise: float = 0.014,
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
        if isinstance(inputs, tf.data.Dataset):
            x = self._tfds_input_perturbation(inputs)
        elif isinstance(inputs, tuple) or isinstance(inputs, np.ndarray):
            x = self._np_input_perturbation(inputs)
        pred = self.feature_extractor(x)[0]
        scores = -np.max(pred, axis=1)
        return scores

    def _tfds_input_perturbation(self, inputs: tf.data.Dataset) -> tf.data.Dataset:
        """
        Compute and apply ODIN's gradient based input perturbation for a tf.data.Dataset

        Parameters
        ----------
        inputs : tf.data.Dataset
            input / dataset with inputs on which to apply the perturbation

        Returns
        -------
        tf.data.Dataset
            Perturbated inputs
        """

        @tf.function
        def input_perturbation(x, num_classes):
            preds = self.feature_extractor.model(x)
            outputs_b = tf.one_hot(tf.argmax(preds, axis=1), num_classes)
            gradients = gradient_single(self.feature_extractor.model, x, outputs_b)
            x = x - self.noise * tf.sign(gradients)
            return x

        num_classes = list(self.feature_extractor.model.layers[-1].output.shape)[1]
        inputs = dataset_get_columns(inputs, 0)
        x = (
            inputs.batch(self.batch_size)
            .map(
                lambda x: input_perturbation(x, num_classes),
            )
            .unbatch()
        )
        return x

    def _np_input_perturbation(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute and apply ODIN's gradient based input perturbation for a np.ndarray

        Parameters
        ----------
        inputs : np.ndarray
            input / dataset with inputs on which to apply the perturbation

        Returns
        -------
        np.ndarray
            Perturbated inputs
        """
        if isinstance(inputs, tuple):
            x = inputs[0]
        else:
            x = inputs
        grads = self.feature_extractor.gradient_pred(x)
        x = x - self.noise * np.sign(-grads)
        return x


def init_softmax_temp(temperature: float = 1):
    """
    Adds softmax_temp to the list of keras activations

    Args:
        temperature: float
    """

    def softmax_temp(inputs):
        return tf.keras.activations.softmax(inputs / temperature)

    return Activation(softmax_temp)
