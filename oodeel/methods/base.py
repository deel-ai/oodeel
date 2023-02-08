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
import torch

from ..models.keras_feature_extractor import KerasFeatureExtractor
from ..models.torch_feature_extractor import TorchFeatureExtractor
from ..types import *
from ..utils.tf_tools import dataset_nb_columns

# TODO find a way to avoid this import and to only import the needed class


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
        output_layers_id: List[int] = [-1],
        input_layers_id: List[int] = 0,
        uses_labels: bool = False,
    ):
        self.feature_extractor = None
        self.output_layers_id = output_layers_id
        self.input_layers_id = input_layers_id
        self.uses_labels = uses_labels

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
        elif isinstance(model, torch.nn.module):
            FeatureExtractor = TorchFeatureExtractor
        else:
            raise NotImplementedError()

        feature_extractor = FeatureExtractor(
            model,
            input_layer_id=self.input_layers_id,
            output_layers_id=self.output_layers_id,
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
        dataset: Union[
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
        # Case 1: dataset is neither a tf.data.Dataset nor a torch.DataLoader
        if isinstance(dataset, np.ndarray) or isinstance(dataset, tf.Tensor):
            return self._score_tensor(dataset)
        elif isinstance(dataset, tuple):
            return (
                self._score_tensor(dataset)
                if self.uses_labels
                else self._score_tensor(dataset[0])
            )
        # Case 2: dataset is a tf.data.Dataset or a torch.DataLoader
        else:
            scores = np.array([])
            assert is_batched(
                dataset
            ), "Please input a batched dataset. Add .batch(batch_size) to your dataset."
            for tensor in dataset:
                if isinstance(tensor, tuple):
                    score_batch = (
                        self._score_tensor(tensor)
                        if self.uses_labels
                        else self._score_tensor(tensor[0])
                    )
                else:
                    score_batch = self._score_tensor(tensor)
                scores = np.append(scores, score_batch)
        return scores

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


def is_batched(dataset):

    nb_column = dataset_nb_columns(dataset)
    if nb_column == 1:
        batch_dim = dataset.element_spec.shape[0]
    else:
        batch_dim = dataset.element_spec[0].shape[0]
    return batch_dim is None
