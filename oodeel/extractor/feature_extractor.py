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

from ..types import Callable
from ..types import DatasetType
from ..types import ItemType
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Tuple
from ..types import Union


class FeatureExtractor(ABC):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers, or the output of the model
    (softmax/logits).

    Args:
        model: model to extract the features from
        feature_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Defaults to [].
        head_layer_id (int, str): identifier of the head layer.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Defaults to -1
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of the
            model).
            Defaults to None.
        react_threshold: if not None, penultimate layer activations are clipped under
            this threshold value (useful for ReAct). Defaults to None.
        scale_percentile: if not None, the features are scaled
            following the method of Xu et al., ICLR 2024.
            Defaults to None.
        ash_percentile: if not None, the features are scaled following
            the method of Djurisic et al., ICLR 2023.
        return_penultimate (bool): if True, the penultimate values are returned,
                i.e. the input to the head_layer.
    """

    def __init__(
        self,
        model: Callable,
        feature_layers_id: List[Union[int, str]] = [],
        head_layer_id: Optional[Union[int, str]] = -1,
        input_layer_id: Optional[Union[int, str]] = [0],
        react_threshold: Optional[float] = None,
        scale_percentile: Optional[float] = None,
        ash_percentile: Optional[float] = None,
        return_penultimate: Optional[bool] = False,
    ):
        if not isinstance(feature_layers_id, list):
            feature_layers_id = [feature_layers_id]

        self.feature_layers_id = feature_layers_id
        self.head_layer_id = head_layer_id
        self.input_layer_id = input_layer_id
        self.react_threshold = react_threshold
        self.scale_percentile = scale_percentile
        self.ash_percentile = ash_percentile
        self.return_penultimate = return_penultimate
        self.model = model
        self.extractor = self.prepare_extractor()

    @abstractmethod
    def prepare_extractor(self) -> None:
        """
        prepare FeatureExtractor for feature extraction
        (the way to achieve this depends on the underlying library)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self, layer_id: Union[str, int]) -> List[TensorType]:
        """
        Get the weights of a layer

        Args:
            layer_id (Union[int, str]): layer identifier

        Returns:
            weights matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_tensor(
        self,
        tensor: TensorType,
        postproc_fns: Optional[List[Callable]] = None,
    ) -> Tuple[List[TensorType], TensorType]:
        """Get the projection of tensor in the feature space of self.model

        Args:
            tensor (TensorType): input tensor (or dataset elem)
            postproc_fns (Optional[Callable]): postprocessing function to apply to each
                feature immediately after forward. Default to None.

        Returns:
            Tuple[List[TensorType], TensorType]: features, logits
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self,
        dataset: Union[ItemType, DatasetType],
        postproc_fns: Optional[List[Callable]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[List[TensorType], dict]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (Union[ItemType, DatasetType]): input dataset
            postproc_fns (Optional[Callable]): postprocessing function to apply to each
                feature immediately after forward. Default to None.
            verbose (bool): if True, display a progress bar. Defaults to False.
            kwargs (dict): additional arguments not considered for prediction

        Returns:
            List[TensorType], dict: features and extra information (logits, labels) as a
                dictionary.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def find_layer(
        model: Callable,
        layer_id: Union[str, int],
        index_offset: int = 0,
        return_id: bool = False,
    ) -> Union[Callable, Tuple[Callable, str]]:
        """Find a layer in a model either by his name or by his index.

        Args:
            model (nn.Module): model whose identified layer will be returned
            layer_id (Union[str, int]): layer identifier
            index_offset (int): index offset to find layers located before (negative
                offset) or after (positive offset) the identified layer
            return_id (bool): if True, the layer will be returned with its id

        Returns:
            Union[Callable, Tuple[Callable, str]]: the corresponding layer and its id if
                return_id is True.
        """
        raise NotImplementedError()

    def __call__(self, inputs: TensorType) -> TensorType:
        """
        Convenience wrapper for predict_tensor().

        Args:
            inputs (Union[DatasetType, TensorType]): input tensor

        Returns:
            TensorType: features
        """
        return self.predict_tensor(inputs)
