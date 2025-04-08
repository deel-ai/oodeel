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
import numpy as np
import keras

from ..types import Callable
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Union
from .general_utils import is_from
from .operator import Operator


if keras.config.backend() == "tensorflow":
    import tensorflow as tf

    Keras3Tensor = tf.Tensor

elif keras.config.backend() == "torch":
    import torch

    Keras3Tensor = torch.Tensor


def sanitize_input(tensor_arg_func: Callable):
    """ensures the decorated function receives a Keras3Tensor"""

    def wrapper(obj, tensor, *args, **kwargs):
        if keras.ops.is_tensor(tensor):
            pass
        elif is_from(tensor, "keras"):
            tensor = keras.ops.convert_to_tensor(tensor.numpy())
        else:
            tensor = keras.ops.convert_to_tensor(tensor)

        return tensor_arg_func(obj, tensor, *args, **kwargs)

    return wrapper


class Keras3Operator(Operator):
    """Class to handle tensorflow operations with a unified API"""

    @staticmethod
    def softmax(tensor: TensorType) -> Keras3Tensor:
        """Softmax function along the last dimension"""
        return keras.activations.softmax(tensor, axis=-1)

    @staticmethod
    def argmax(tensor: TensorType, dim: Optional[int] = None) -> Keras3Tensor:
        """Argmax function"""
        return keras.ops.argmax(tensor, axis=dim)

    @staticmethod
    def max(
        tensor: TensorType, dim: Optional[int] = None, keepdim: bool = False
    ) -> Keras3Tensor:
        """Max function"""
        return keras.ops.max(tensor, axis=dim, keepdims=keepdim)

    @staticmethod
    def min(
        tensor: TensorType, dim: Optional[int] = None, keepdim: bool = False
    ) -> Keras3Tensor:
        """Min function"""
        return keras.ops.min(tensor, axis=dim, keepdims=keepdim)

    @staticmethod
    def one_hot(tensor: TensorType, num_classes: int) -> Keras3Tensor:
        """One hot function"""
        return keras.ops.one_hot(tensor, num_classes)

    @staticmethod
    def sign(tensor: TensorType) -> Keras3Tensor:
        """Sign function"""
        return keras.ops.sign(tensor)

    @staticmethod
    def CrossEntropyLoss(reduction: str = "mean"):
        """Cross Entropy Loss from logits"""

        def sanitized_ce_loss(inputs, targets):
            return keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=reduction
            )(targets, inputs)

        return sanitized_ce_loss

    @staticmethod
    def norm(tensor: TensorType, dim: Optional[int] = None) -> Keras3Tensor:
        """Tensor Norm"""
        return keras.ops.norm(tensor, axis=dim)

    @staticmethod
    def matmul(tensor_1: TensorType, tensor_2: TensorType) -> Keras3Tensor:
        """Matmul operation"""
        return keras.ops.matmul(tensor_1, tensor_2)

    @staticmethod
    def convert_to_numpy(tensor: TensorType) -> np.ndarray:
        """Convert tensor into a np.ndarray"""
        try:
            return tensor.numpy()
        except TypeError:  # if the tensor is a torch tensor on GPU, put it on CPU first
            if tensor.device != "cpu":
                tensor = tensor.to("cpu")
            return tensor.detach().numpy()

    @staticmethod
    def gradient(func: Callable, inputs: Keras3Tensor, *args, **kwargs) -> Keras3Tensor:
        """Compute gradients for a batch of samples.

        Args:
            func (Callable): Function used for computing gradient. Must be built with
                tensorflow differentiable operations only, and return a scalar.
            inputs (Keras3Tensor): Input tensor wrt which the gradients are computed
            *args: Additional Args for func.
            **kwargs: Additional Kwargs for func.

        Returns:
            Keras3Tensor: Gradients computed, with the same shape as the inputs.
        """
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(inputs)
                outputs = func(inputs, *args, **kwargs)
            return tape.gradient(outputs, inputs)
        elif keras.config.backend() == "torch":
            import torch

            inputs.requires_grad_(True)
            outputs = func(inputs, *args, **kwargs)
            gradients = torch.autograd.grad(outputs, inputs)
            inputs.requires_grad_(False)
            return gradients[0]

    @staticmethod
    def stack(tensors: List[TensorType], dim: int = 0) -> Keras3Tensor:
        "Stack tensors along a new dimension"
        return keras.ops.stack(tensors, dim)

    @staticmethod
    def cat(tensors: List[TensorType], dim: int = 0) -> Keras3Tensor:
        "Concatenate tensors in a given dimension"
        return keras.ops.concatenate(tensors, dim)

    @staticmethod
    def mean(tensor: TensorType, dim: Optional[int] = None) -> Keras3Tensor:
        "Mean function"
        return keras.ops.mean(tensor, dim)

    @staticmethod
    def flatten(tensor: TensorType) -> Keras3Tensor:
        "Flatten to 2D tensor of shape (tensor.shape[0], -1)"
        # Flatten the features to 2D (n_batch, n_features)
        return keras.ops.reshape(tensor, newshape=[keras.ops.shape(tensor)[0], -1])

    @staticmethod
    def from_numpy(arr: np.ndarray) -> Keras3Tensor:
        "Convert a NumPy array to a tensor"
        # TODO change dtype
        return keras.convert_to_tensor(arr)

    @staticmethod
    def t(tensor: TensorType) -> Keras3Tensor:
        "Transpose function for tensor of rank 2"
        return keras.ops.transpose(tensor)

    @staticmethod
    def permute(tensor: TensorType, dims) -> Keras3Tensor:
        "Transpose function for tensor of rank 2"
        return keras.ops.transpose(tensor, dims)

    @staticmethod
    def diag(tensor: TensorType) -> Keras3Tensor:
        "Diagonal function: return the diagonal of a 2D tensor"
        return keras.ops.diag(tensor)

    @staticmethod
    def reshape(tensor: TensorType, shape: List[int]) -> Keras3Tensor:
        "Reshape function"
        return keras.ops.reshape(tensor, shape)

    @staticmethod
    def equal(tensor: TensorType, other: Union[TensorType, int, float]) -> Keras3Tensor:
        "Computes element-wise equality"
        return keras.ops.equal(tensor, other)

    @staticmethod
    def pinv(tensor: TensorType) -> Keras3Tensor:
        "Computes the pseudoinverse (Moore-Penrose inverse) of a matrix."
        if keras.config.backend() == "tensorflow":
            import tensorflow as tf

            return tf.linalg.pinv(tensor)
        elif keras.config.backend() == "torch":
            import torch

            return torch.linalg.pinv(tensor)

    @staticmethod
    def eigh(tensor: TensorType) -> Keras3Tensor:
        "Computes the eigen decomposition of a self-adjoint matrix."
        eigval, eigvec = keras.ops.eigh(tensor)
        return eigval, eigvec

    @staticmethod
    def quantile(tensor: TensorType, q: float, dim: int = None) -> Keras3Tensor:
        "Computes the quantile of a tensor's components. q in (0,1)"
        return keras.ops.quantile(tensor, q, dim)

    @staticmethod
    def relu(tensor: TensorType) -> Keras3Tensor:
        "Apply relu to a tensor"
        return keras.ops.relu(tensor)

    @staticmethod
    def einsum(equation: str, *tensors: TensorType) -> Keras3Tensor:
        "Computes the einsum between tensors following equation"
        return keras.ops.einsum(equation, *tensors)

    @staticmethod
    def tril(tensor: TensorType, diagonal: int = 0) -> Keras3Tensor:
        "Set the upper triangle of the matrix formed by the last two dimensions of"
        "tensor to zero"
        return keras.ops.tril(tensor, k=diagonal)

    @staticmethod
    def sum(tensor: TensorType, dim: Union[tuple, list, int] = None) -> Keras3Tensor:
        "sum along dim"
        return keras.ops.sum(tensor, axis=dim)

    @staticmethod
    def unsqueeze(tensor: TensorType, dim: int) -> Keras3Tensor:
        "expand_dim along dim"
        return keras.ops.expand_dims(tensor, dim)

    @staticmethod
    def squeeze(tensor: TensorType, dim: int = None) -> Keras3Tensor:
        "expand_dim along dim"
        return keras.ops.squeeze(tensor, dim)

    @staticmethod
    def abs(tensor: TensorType) -> Keras3Tensor:
        "compute absolute value"
        return keras.ops.abs(tensor)

    @staticmethod
    def where(
        condition: TensorType,
        input: Union[TensorType, float],
        other: Union[TensorType, float],
    ) -> Keras3Tensor:
        "Applies where function to condition"
        return keras.ops.where(condition, input, other)

    @staticmethod
    def avg_pool_2d(tensor: TensorType) -> Keras3Tensor:
        """Perform avg pool in 2d as in torch.nn.functional.adaptive_avg_pool2d"""
        # TODO: axis should be different depending on backend
        return keras.ops.mean(tensor, axis=(-3, -2))

    @staticmethod
    def log(tensor: TensorType) -> Keras3Tensor:
        """Perform log"""
        return keras.ops.log(tensor)
