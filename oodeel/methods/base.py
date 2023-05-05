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
from typing import get_args

import numpy as np

from ..types import Callable
from ..types import DatasetType
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Union
from ..utils import is_from


class OODModel(ABC):
    """Base Class for methods that assign a score to unseen samples.

    Args:
        output_layers_id (List[int]): list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer. Defaults to [-1],
        input_layers_id (List[int]): = list of str or int that identify the input layer
            of the feature extractor.
            If int, the rank of the layer in the layer list
            If str, the name of the layer. Defaults to None.
    """

    def __init__(
        self,
        output_layers_id: List[int] = [-1],
        input_layers_id: List[int] = None,
    ):
        self.feature_extractor = None
        self.output_layers_id = output_layers_id
        self.input_layers_id = input_layers_id

    @abstractmethod
    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """Computes an OOD score for input samples "inputs".
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
        fit_dataset: Optional[Union[TensorType, DatasetType]] = None,
    ) -> None:
        """Prepare oodmodel for scoring:
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
        if is_from(model, "keras"):
            from ..models.keras_feature_extractor import KerasFeatureExtractor
            from ..datasets.tf_data_handler import TFDataHandler
            from ..utils import TFOperator

            self.data_handler = TFDataHandler()
            self.op = TFOperator()
            self.backend = "tensorflow"
            FeatureExtractor = KerasFeatureExtractor

        elif is_from(model, "torch"):
            from ..models.torch_feature_extractor import TorchFeatureExtractor
            from ..datasets.torch_data_handler import TorchDataHandler
            from ..utils import TorchOperator

            self.data_handler = TorchDataHandler()
            self.op = TorchOperator(model)
            self.backend = "torch"
            FeatureExtractor = TorchFeatureExtractor

        else:
            raise NotImplementedError()

        feature_extractor = FeatureExtractor(
            model,
            input_layer_id=self.input_layers_id,
            output_layers_id=self.output_layers_id,
        )
        return feature_extractor

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]):
        """
        Fits the oodmodel to fit_dataset.
        To be overrided in child classes (if needed)

        Args:
            fit_dataset: dataset to fit the oodmodel on
        """
        raise NotImplementedError()

    def calibrate_threshold(
        self,
        fit_dataset: Union[TensorType, DatasetType],
        scores: np.ndarray,
    ) -> None:
        """
        Calibrates the model on ID data "id_dataset".
        Placeholder for now

        Args:
            fit_dataset: dataset to callibrate the threshold on
            scores: scores of oodmodel
        """
        raise NotImplementedError()

    def score(
        self,
        dataset: Union[TensorType, DatasetType],
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs"

        Args:
            dataset (Union[TensorType, DatasetType]): dataset or tensors to score

        Returns:
            scores or list of scores (depending on the input)
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"

        # Case 1: dataset is neither a tf.data.Dataset nor a torch.DataLoader
        if isinstance(dataset, get_args(TensorType)):
            tensor = self.data_handler.get_input_from_dataset_item(dataset)
            scores = self._score_tensor(tensor)
        # Case 2: dataset is a tf.data.Dataset or a torch.DataLoader
        elif isinstance(dataset, get_args(DatasetType)):
            scores = np.array([])
            for tensor in dataset:
                tensor = self.data_handler.get_input_from_dataset_item(tensor)
                score_batch = self._score_tensor(tensor)
                scores = np.append(scores, score_batch)
        else:
            raise NotImplementedError(
                f"OODModel.score() not implemented for {type(dataset)}"
            )
        return scores

    def isood(
        self, dataset: Union[TensorType, DatasetType], threshold: float
    ) -> np.ndarray:
        """
        Returns whether the input samples "inputs" are OOD or not, given a threshold

        Args:
            dataset (dataset: Union[TensorType, DatasetType]): dataset or tensors to score
            threshold (float): threshold to use for distinguishing between OOD and ID

        Returns:
            np.ndarray: array of 0 for ID samples and 1 for OOD samples
        """
        assert self.feature_extractor is not None, "Call .fit() before .isood()"

        # Case 1: dataset is neither a tf.data.Dataset nor a torch.DataLoader
        if isinstance(dataset, get_args(TensorType)):
            tensor = self.data_handler.get_input_from_dataset_item(dataset)
            scores = self._score_tensor(tensor)
        # Case 2: dataset is a tf.data.Dataset or a torch.DataLoader
        elif isinstance(dataset, get_args(DatasetType)):
            scores = np.array([])
            for tensor in dataset:
                tensor = self.data_handler.get_input_from_dataset_item(tensor)
                score_batch = self._score_tensor(tensor)
                scores = np.append(scores, score_batch)
        else:
            raise NotImplementedError(
                f"OODModel.isood() not implemented for {type(dataset)}"
            )
        oodness = scores < threshold
        return np.array(oodness, dtype=np.bool)

    def __call__(
        self, inputs: Union[TensorType, DatasetType], threshold: float
    ) -> np.ndarray:
        """
        Convenience wrapper for isood
        """
        return self.isood(inputs, threshold)
