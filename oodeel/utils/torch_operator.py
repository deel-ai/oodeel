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
from typing import List

import numpy as np
import torch

from ..types import Callable
from ..types import TensorType
from ..types import Union
from .general_utils import is_from
from .operator import Operator


def sanitize_input(tensor_arg_func: Callable):
    def wrapper(obj, tensor, *args, **kwargs):
        if isinstance(tensor, torch.Tensor):
            pass
        elif is_from(tensor, "tensorflow"):
            tensor = torch.Tensor(tensor.numpy())
        else:
            tensor = torch.Tensor(tensor)

        return tensor_arg_func(obj, tensor, *args, **kwargs)

    return wrapper


class TorchOperator(Operator):
    """Class to handle torch operations with a unified API"""

    @staticmethod
    def softmax(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Softmax function"""
        return torch.nn.functional.softmax(tensor)

    @staticmethod
    def argmax(
        tensor: Union[torch.Tensor, np.ndarray], dim: int = None
    ) -> torch.Tensor:
        """Argmax function"""
        return torch.argmax(tensor, dim=dim)

    @staticmethod
    def max(tensor: Union[torch.Tensor, np.ndarray], dim: int = None) -> torch.Tensor:
        """Max function"""
        return torch.max(tensor, dim=dim)

    @staticmethod
    def one_hot(
        tensor: Union[torch.Tensor, np.ndarray], num_classes: int
    ) -> torch.Tensor:
        """One hot function"""
        return torch.nn.functional.one_hot(tensor, num_classes)

    @staticmethod
    def sign(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Sign function"""
        return torch.sign(tensor)

    @staticmethod
    def CrossEntropyLoss(reduction: str = "mean"):
        """Cross Entropy Loss from logits"""

        def sanitized_ce_loss(inputs, targets):
            return torch.nn.CrossEntropyLoss(reduction=reduction)(inputs, targets)

        return sanitized_ce_loss

    @staticmethod
    def norm(tensor: Union[torch.Tensor, np.ndarray], dim: int = None) -> torch.Tensor:
        """Tensor Norm"""
        return tensor.norm(dim=dim)

    @staticmethod
    def matmul(tensor_1: TensorType, tensor_2: TensorType) -> TensorType:
        """Matmul operation"""
        return torch.matmul(tensor_1, tensor_2)

    @staticmethod
    def convert_from_tensorflow(tensor: TensorType) -> torch.Tensor:
        """Convert a tensorflow tensor into a torch tensor

        Used when using a pytorch model on a dataset loaded from tensorflow datasets
        """
        return torch.Tensor(tensor.numpy())

    @staticmethod
    def convert_to_numpy(tensor: TensorType) -> np.ndarray:
        if tensor.device != "cpu":
            tensor = tensor.to("cpu")
        return tensor.detach().numpy()

    @staticmethod
    def gradient(func: Callable, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute gradients for a batch of samples.

        Args:
            func (Callable): Function used for computing gradient. Must be built with
                torch differentiable operations only, and return a scalar.
            inputs (torch.Tensor): Input tensor wrt which the gradients are computed

        Returns:
            torch.Tensor: Gradients computed, with the same shape as the inputs.
        """
        inputs.requires_grad_(True)
        outputs = func(inputs, *args, **kwargs)
        gradients = torch.autograd.grad(outputs, inputs)
        inputs.requires_grad_(False)
        return gradients[0]

    @staticmethod
    def stack(tensors: List[TensorType], dim: int = 0) -> TensorType:
        "Stack tensors along a new dimension"
        return torch.stack(tensors, dim)

    @staticmethod
    def cat(tensors: List[TensorType], dim: int = 0) -> TensorType:
        "Concatenate tensors in a given dimension"
        return torch.cat(tensors, dim)

    @staticmethod
    def mean(tensor: TensorType, dim: int = None, keepdim: bool = False) -> TensorType:
        "Mean function"
        return torch.mean(tensor, dim, keepdim)

    @staticmethod
    def flatten(tensor: TensorType) -> TensorType:
        "Flatten function"
        # Flatten the features to 2D (n_batch, n_features)
        return tensor.view(tensor.size(0), -1)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> TensorType:
        "Convert a NumPy array to a tensor"
        # TODO change dtype
        return torch.from_numpy(arr).double()

    @staticmethod
    def transpose(tensor: TensorType) -> TensorType:
        "Transpose function for tensor of rank 2"
        return tensor.t()

    @staticmethod
    def diag(tensor: TensorType) -> TensorType:
        "Diagonal function: return the diagonal of a 2D tensor"
        return tensor.diag()

    @staticmethod
    def reshape(tensor: TensorType, shape: List[int]) -> TensorType:
        "Reshape function"
        return tensor.view(*shape)
