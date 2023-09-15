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
from typing import Optional

import numpy as np
import torch

from ..types import Callable
from ..types import TensorType
from ..types import Union
from .general_utils import is_from
from .operator import Operator


def sanitize_input(tensor_arg_func: Callable):
    """ensures the decorated function receives a torch.Tensor"""

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

    def __init__(self, model: Optional[torch.nn.Module] = None):
        if model is not None:
            self._device = next(model.parameters()).device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def softmax(tensor: TensorType) -> torch.Tensor:
        """Softmax function along the last dimension"""
        return torch.nn.functional.softmax(tensor, dim=-1)

    @staticmethod
    def argmax(tensor: TensorType, dim: Optional[int] = None) -> torch.Tensor:
        """Argmax function"""
        return torch.argmax(tensor, dim=dim)

    @staticmethod
    def max(
        tensor: TensorType, dim: Optional[int] = None, keepdim: Optional[bool] = False
    ) -> torch.Tensor:
        """Max function"""
        if dim is None:
            return torch.max(tensor)
        else:
            return torch.max(tensor, dim, keepdim=keepdim)[0]

    @staticmethod
    def min(
        tensor: TensorType, dim: Optional[int] = None, keepdim: bool = False
    ) -> torch.Tensor:
        """Min function"""
        if dim is None:
            return torch.min(tensor)
        else:
            return torch.min(tensor, dim, keepdim=keepdim)[0]

    @staticmethod
    def one_hot(tensor: TensorType, num_classes: int) -> torch.Tensor:
        """One hot function"""
        return torch.nn.functional.one_hot(tensor, num_classes)

    @staticmethod
    def sign(tensor: TensorType) -> torch.Tensor:
        """Sign function"""
        return torch.sign(tensor)

    @staticmethod
    def CrossEntropyLoss(reduction: str = "mean"):
        """Cross Entropy Loss from logits"""

        def sanitized_ce_loss(inputs, targets):
            return torch.nn.CrossEntropyLoss(reduction=reduction)(inputs, targets)

        return sanitized_ce_loss

    @staticmethod
    def norm(tensor: TensorType, dim: Optional[int] = None) -> torch.Tensor:
        """Tensor Norm"""
        return torch.norm(tensor, dim=dim)

    @staticmethod
    def matmul(tensor_1: TensorType, tensor_2: TensorType) -> torch.Tensor:
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
        """Convert tensor into a np.ndarray"""
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
            *args: Additional Args for func.
            **kwargs: Additional Kwargs for func.

        Returns:
            torch.Tensor: Gradients computed, with the same shape as the inputs.
        """
        inputs.requires_grad_(True)
        outputs = func(inputs, *args, **kwargs)
        gradients = torch.autograd.grad(outputs, inputs)
        inputs.requires_grad_(False)
        return gradients[0]

    @staticmethod
    def stack(tensors: List[TensorType], dim: int = 0) -> torch.Tensor:
        "Stack tensors along a new dimension"
        return torch.stack(tensors, dim)

    @staticmethod
    def cat(tensors: List[TensorType], dim: int = 0) -> torch.Tensor:
        "Concatenate tensors in a given dimension"
        return torch.cat(tensors, dim)

    @staticmethod
    def mean(tensor: TensorType, dim: Optional[int] = None) -> torch.Tensor:
        "Mean function"
        if dim is None:
            return torch.mean(tensor)
        else:
            return torch.mean(tensor, dim)

    @staticmethod
    def flatten(tensor: TensorType) -> torch.Tensor:
        "Flatten function"
        # Flatten the features to 2D (n_batch, n_features)
        return tensor.view(tensor.size(0), -1)

    def from_numpy(self, arr: np.ndarray) -> torch.Tensor:
        "Convert a NumPy array to a tensor"
        # TODO change dtype
        return torch.tensor(arr).to(self._device)

    @staticmethod
    def t(tensor: TensorType) -> torch.Tensor:
        "Transpose function for tensor of rank 2"
        return tensor.t()

    @staticmethod
    def permute(tensor: TensorType, dims) -> torch.Tensor:
        "Transpose function for tensor of rank 2"
        return torch.permute(tensor, dims)

    @staticmethod
    def diag(tensor: TensorType) -> torch.Tensor:
        "Diagonal function: return the diagonal of a 2D tensor"
        return tensor.diag()

    @staticmethod
    def reshape(tensor: TensorType, shape: List[int]) -> torch.Tensor:
        "Reshape function"
        return tensor.view(*shape)

    @staticmethod
    def equal(tensor: TensorType, other: Union[TensorType, int, float]) -> torch.Tensor:
        "Computes element-wise equality"
        return torch.eq(tensor, other)

    @staticmethod
    def pinv(tensor: TensorType) -> torch.Tensor:
        "Computes the pseudoinverse (Moore-Penrose inverse) of a matrix."
        return torch.linalg.pinv(tensor)

    @staticmethod
    def eigh(tensor: TensorType) -> torch.Tensor:
        "Computes the eigen decomposition of a self-adjoint matrix."
        eigval, eigvec = torch.linalg.eigh(tensor)
        return eigval, eigvec

    @staticmethod
    def quantile(tensor: TensorType, q: float, dim: int = None) -> torch.Tensor:
        "Computes the quantile of a tensor's components. q in (0,1)"
        if dim is None:
            # keep the 16 millions first elements (see torch.quantile issue:
            # https://github.com/pytorch/pytorch/issues/64947)
            tensor_flatten = tensor.view(-1)[:16_000_000]
            return torch.quantile(tensor_flatten, q).item()
        else:
            return torch.quantile(tensor, q, dim)

    @staticmethod
    def relu(tensor: TensorType) -> torch.Tensor:
        "Apply relu to a tensor"
        return torch.nn.functional.relu(tensor)

    @staticmethod
    def einsum(equation: str, *tensors: TensorType) -> torch.Tensor:
        "Computes the einsum between tensors following equation"
        return torch.einsum(equation, *tensors)

    @staticmethod
    def tril(tensor: TensorType, diagonal: int = 0) -> torch.Tensor:
        "Set the upper triangle of the matrix formed by the last two dimensions of"
        "tensor to zero"
        return torch.tril(tensor, diagonal)

    @staticmethod
    def sum(tensor: TensorType, dim: Union[tuple, list, int] = None) -> torch.Tensor:
        "sum along dim"
        return torch.sum(tensor, dim)

    @staticmethod
    def unsqueeze(tensor: TensorType, dim: int) -> torch.Tensor:
        "unsqueeze along dim"
        return torch.unsqueeze(tensor, dim)

    @staticmethod
    def abs(tensor: TensorType) -> torch.Tensor:
        "compute absolute value"
        return torch.abs(tensor)

    @staticmethod
    def where(
        condition: TensorType,
        input: Union[TensorType, float],
        other: Union[TensorType, float],
    ) -> torch.Tensor:
        "Applies where function , to condition"
        return torch.where(condition, input, other)
