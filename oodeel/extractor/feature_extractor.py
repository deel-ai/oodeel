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

from ..types import Any
from ..types import Callable
from ..types import List
from ..types import Union


class FeatureExtractor(ABC):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers, or the output of the model
    (softmax/logits).

    Args:
        model: model to extract the features from
        output_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Defaults to [].
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of the
            model).
            Defaults to None.
    """

    def __init__(
        self,
        model: Callable,
        output_layers_id: List[Union[int, str]] = [-1],
        input_layer_id: Union[int, str] = [0],
    ):
        if not isinstance(output_layers_id, list):
            output_layers_id = [output_layers_id]

        self.output_layers_id = output_layers_id
        self.input_layer_id = input_layer_id
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
    def get_weights(self) -> List[Any]:
        """
        Get the weights of a layer

        Returns:
            weights matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_tensor(self, tensor: Any) -> Any:
        """
        Projects input samples "inputs" into the feature space

        Args:
            tensor (Any): input tensor

        Returns:
            Any: features
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataset: Any) -> Any:
        """
        Projects input samples "inputs" into the feature space for a batched dataset

        Args:
            dataset (Any): iterable of tensor batches

        Returns:
            Any: features
        """
        raise NotImplementedError()

    def __call__(self, inputs: Any) -> Any:
        """
        Convenience wrapper for predict_tensor().

        Args:
            inputs (Any): input tensor

        Returns:
            Any: features
        """
        return self.predict_tensor(inputs)
