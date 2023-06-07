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

import numpy as np

from ..types import Callable
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Union


class Operator(ABC):
    """Class to handle tensorflow and torch operations with a unified API"""

    @staticmethod
    @abstractmethod
    def softmax(tensor: TensorType) -> TensorType:
        """Softmax function along the last dimension"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def argmax(tensor: TensorType, dim: Optional[int] = None) -> TensorType:
        """Argmax function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def max(tensor: TensorType, dim: Optional[int] = None) -> TensorType:
        """Max function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def one_hot(tensor: TensorType, num_classes: int) -> TensorType:
        """One hot function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def sign(tensor: TensorType) -> TensorType:
        """Sign function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def CrossEntropyLoss(reduction: str = "mean"):
        """Cross Entropy Loss from logits"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def norm(tensor: TensorType, dim: Optional[int] = None) -> TensorType:
        """Norm function"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def matmul(tensor_1: TensorType, tensor_2: TensorType) -> TensorType:
        """Matmul operation"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def convert_to_numpy(tensor: TensorType) -> np.ndarray:
        "Convert a tensor to a NumPy array"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def gradient(func: Callable, inputs: TensorType) -> TensorType:
        """Compute gradients for a batch of samples.

        Args:
            func (Callable): Function used for computing gradient. Must be built with
                differentiable operations only, and return a scalar.
            inputs (Any): Input tensor wrt which the gradients are computed

        Returns:
            Gradients computed, with the same shape as the inputs.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def stack(tensors: List[TensorType], dim: int = 0) -> TensorType:
        "Stack tensors along a new dimension"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def cat(tensors: List[TensorType], dim: int = 0) -> TensorType:
        "Concatenate tensors in a given dimension"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def mean(tensor: TensorType, dim: Optional[int] = None) -> TensorType:
        "Mean function"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def flatten(tensor: TensorType) -> TensorType:
        "Flatten to 2D tensor (batch_size, -1)"
        # Flatten the features to 2D (n_batch, n_features)
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_numpy(arr: np.ndarray) -> TensorType:
        "Convert a NumPy array to a tensor"
        # TODO change dtype
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def transpose(tensor: TensorType) -> TensorType:
        "Transpose function for tensor of rank 2"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def diag(tensor: TensorType) -> TensorType:
        "Diagonal function: return the diagonal of a 2D tensor"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def reshape(tensor: TensorType, shape: List[int]) -> TensorType:
        "Reshape function"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def equal(tensor: TensorType, other: Union[TensorType, int, float]) -> TensorType:
        "Computes element-wise equality"
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def pinv(tensor: TensorType) -> TensorType:
        "Computes the pseudoinverse (Moore-Penrose inverse) of a matrix."
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def eigh(tensor: TensorType) -> TensorType:
        "Computes the eigen decomposition of a self-adjoint matrix."
        raise NotImplementedError()

    @staticmethod
    def quantile(tensor: TensorType, q: float, dim: int = None) -> TensorType:
        "Computes the quantile of a tensor's components"
        raise NotImplementedError()

    @staticmethod
    def relu(tensor: TensorType) -> TensorType:
        "Apply relu to a tensor"
        raise NotImplementedError()
