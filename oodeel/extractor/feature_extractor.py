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
from ..types import List
from ..types import Optional
from ..types import TensorType
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
        postproc_fns: Optional[List[Callable]] = None,
    ):
        if not isinstance(output_layers_id, list):
            output_layers_id = [output_layers_id]

        self.output_layers_id = output_layers_id
        self.input_layer_id = input_layer_id
        self.model = model
        self.postproc_fns = self.sanitize_posproc_fns(postproc_fns)
        self.extractor = self.prepare_extractor()

    def sanitize_posproc_fns(
        self,
        postproc_fns: Union[List[Callable], None],
    ) -> List[Callable]:
        """Sanitize postproc fns used at each layer output of the feature extractor.

        Args:
            postproc_fns (Optional[List[Callable]], optional): List of postproc functions,
                one per output layer. Defaults to None.

        Returns:
            List[Callable]: Sanitized postproc_fns list
        """
        if postproc_fns is not None:
            assert len(postproc_fns) == len(
                self.output_layers_id
            ), "len of postproc_fns and output_layers_id must match"

            def identity(x):
                return x

            postproc_fns = [identity if fn is None else fn for fn in postproc_fns]

        return postproc_fns

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
    def predict_tensor(self, tensor: TensorType) -> Union[TensorType, List[TensorType]]:
        """
        Projects input samples "inputs" into the feature space

        Args:
            tensor (TensorType): input tensor

        Returns:
            TensorType: features
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self,
        dataset: Union[DatasetType, TensorType],
        postproc_fn: Optional[Callable] = None,
    ) -> Union[TensorType, List[TensorType]]:
        """
        Projects input samples "inputs" into the feature space for a batched dataset

        Args:
            dataset (Union[DatasetType, TensorType]): iterable of tensor batches

        Returns:
            TensorType: features
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
