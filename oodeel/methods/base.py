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
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Iterable
from typing import Type
from typing import Union

import numpy as np
import tensorflow as tf

from ..models.keras_feature_extractor import KerasFeatureExtractor
from ..types import *


class OODModel(ABC):
    """
    Base Class for methods that assign a score to unseen samples.

    Args:
        output_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Defaults to [].
        output_activation: activation function for the last layer.
            Defaults to None.
        flatten: Flatten the output features or not.
            Defaults to True.
        batch_size: batch_size used to compute the features space
            projection of input data.
            Defaults to 256.
    """

    def __init__(
        self,
        output_layers_id: List[int] = [],
        output_activation: str = None,
        flatten: bool = False,
        batch_size: int = 256,
    ):

        self.batch_size = batch_size
        self.feature_extractor = None
        self.output_layers_id = output_layers_id
        self.output_activation = output_activation
        self.flatten = flatten

    @abstractmethod
    def _score_tensor(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """
        Computes an OOD score for input samples "inputs".
        Method to override with child classes.

        Args:
            inputs: tensor to score

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def fit(
        self,
        model: Callable,
        fit_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
    ):
        """
        Prepare oodmodel for scoring:
        * Constructs the feature extractor based on the model
        * Calibrates the oodmodel on ID data "fit_dataset" if needed,
            using self._fit_to_dataset

        Args:
            model: model to extract the features from
            fit_dataset: dataset to fit the oodmodel on
        """
        self.feature_extractor = self._load_feature_extractor(model)
        if fit_dataset is not None:
            self._fit_to_dataset(fit_dataset)

    def _load_feature_extractor(
        self,
        model: Callable,
    ) -> Callable:
        """
        Loads feature extractor

        Args:
            model : tf.keras model (for now)
                keras models saved as pb files e.g. with model.save()
        """
        if isinstance(model, tf.keras.Model):
            FeatureExtractor = KerasFeatureExtractor
        else:
            raise NotImplementedError()

        feature_extractor = FeatureExtractor(
            model,
            output_layers_id=self.output_layers_id,
            output_activation=self.output_activation,
            flatten=self.flatten,
            batch_size=self.batch_size,
        )
        return feature_extractor

    def _fit_to_dataset(
        self, fit_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ):
        """
        Fits the oodmodel to fit_dataset.
        To be overrided in child classes (if needed)

        Args:
            fit_dataset: dataset to fit the oodmodel on

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def calibrate_threshold(
        self,
        fit_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        scores: np.ndarray,
    ):
        """
        Calibrates the model on ID data "id_dataset".
        Placeholder for now

        Args:
            fit_dataset: dataset to callibrate the threshold on
            scores: scores of oodmodel

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def score(
        self,
        inputs: Union[
            List[Union[tf.data.Dataset, tf.Tensor, np.ndarray]],
            Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        ],
        outputs: Optional[
            Union[
                List[Union[tf.data.Dataset, tf.Tensor, np.ndarray]],
                Union[tf.data.Dataset, tf.Tensor, np.ndarray],
            ]
        ] = None,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Computes an OOD score for input samples "inputs"

        Args:
            inputs: Tensors, or list of tensors to score

        Returns:
            scores or list of scores (depending on the input)
        """
        if type(inputs) is not list:
            scores = self._score_tensor(inputs)
            return scores
        else:
            scores_list = []
            for input in inputs:
                scores = (
                    self._score_tensor(input)
                    if outputs is None
                    else self._score_tensor(input, outputs)
                )
                scores_list.append(scores)
            return scores_list

    def isood(
        self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray], threshold: float
    ) -> np.ndarray:
        """
        Returns whether the input samples "inputs" are OOD or not, given a threshold

        Args:
            inputs: input samples to score
            threshold: threshold to use for distinguishing between OOD and ID

        Returns:
            np.array of 0 for ID samples and 1 for OOD samples
        """
        scores = self.score(inputs)
        OODness = tf.map_fn(lambda x: 0 if x < threshold else 1, scores)

        return OODness

    def __call__(
        self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray], threshold: float
    ) -> np.ndarray:
        """
        Convenience wrapper for isood
        """
        return self.isood(inputs, threshold)
