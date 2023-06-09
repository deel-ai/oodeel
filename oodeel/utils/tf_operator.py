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
import tensorflow as tf
import tensorflow_probability as tfp

from ..types import Callable
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Union
from .general_utils import is_from
from .operator import Operator


def sanitize_input(tensor_arg_func: Callable):
    """ensures the decorated function receives a tf.Tensor"""

    def wrapper(obj, tensor, *args, **kwargs):
        if isinstance(tensor, tf.Tensor):
            pass
        elif is_from(tensor, "torch"):
            tensor = tf.convert_to_tensor(tensor.numpy())
        else:
            tensor = tf.convert_to_tensor(tensor)

        return tensor_arg_func(obj, tensor, *args, **kwargs)

    return wrapper


class TFOperator(Operator):
    """Class to handle tensorflow operations with a unified API"""

    @staticmethod
    def softmax(tensor: TensorType) -> tf.Tensor:
        """Softmax function along the last dimension"""
        return tf.keras.activations.softmax(tensor, axis=-1)

    @staticmethod
    def argmax(tensor: TensorType, dim: Optional[int] = None) -> tf.Tensor:
        """Argmax function"""
        if dim is None:
            return tf.argmax(tf.reshape(tensor, [-1]))
        return tf.argmax(tensor, axis=dim)

    @staticmethod
    def max(tensor: TensorType, dim: Optional[int] = None) -> tf.Tensor:
        """Max function"""
        return tf.reduce_max(tensor, axis=dim)

    @staticmethod
    def one_hot(tensor: TensorType, num_classes: int) -> tf.Tensor:
        """One hot function"""
        return tf.one_hot(tensor, num_classes)

    @staticmethod
    def sign(tensor: TensorType) -> tf.Tensor:
        """Sign function"""
        return tf.sign(tensor)

    @staticmethod
    def CrossEntropyLoss(reduction: str = "mean"):
        """Cross Entropy Loss from logits"""

        tf_reduction = {"mean": "sum_over_batch_size", "sum": "sum"}[reduction]

        def sanitized_ce_loss(inputs, targets):
            return tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf_reduction
            )(targets, inputs)

        return sanitized_ce_loss

    @staticmethod
    def norm(tensor: TensorType, dim: Optional[int] = None) -> tf.Tensor:
        """Tensor Norm"""
        return tf.norm(tensor, axis=dim)

    @staticmethod
    @tf.function
    def matmul(tensor_1: TensorType, tensor_2: TensorType) -> tf.Tensor:
        """Matmul operation"""
        return tf.matmul(tensor_1, tensor_2)

    @staticmethod
    def convert_to_numpy(tensor: TensorType) -> np.ndarray:
        """Convert tensor into a np.ndarray"""
        return tensor.numpy()

    @staticmethod
    def gradient(func: Callable, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Compute gradients for a batch of samples.

        Args:
            func (Callable): Function used for computing gradient. Must be built with
                tensorflow differentiable operations only, and return a scalar.
            inputs (tf.Tensor): Input tensor wrt which the gradients are computed
            *args: Additional Args for func.
            **kwargs: Additional Kwargs for func.

        Returns:
            tf.Tensor: Gradients computed, with the same shape as the inputs.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            outputs = func(inputs, *args, **kwargs)
        return tape.gradient(outputs, inputs)

    @staticmethod
    def stack(tensors: List[TensorType], dim: int = 0) -> tf.Tensor:
        "Stack tensors along a new dimension"
        return tf.stack(tensors, dim)

    @staticmethod
    def cat(tensors: List[TensorType], dim: int = 0) -> tf.Tensor:
        "Concatenate tensors in a given dimension"
        return tf.concat(tensors, dim)

    @staticmethod
    def mean(tensor: TensorType, dim: Optional[int] = None) -> tf.Tensor:
        "Mean function"
        return tf.reduce_mean(tensor, dim)

    @staticmethod
    def flatten(tensor: TensorType) -> tf.Tensor:
        "Flatten to 2D tensor of shape (tensor.shape[0], -1)"
        # Flatten the features to 2D (n_batch, n_features)
        return tf.reshape(tensor, shape=[tf.shape(tensor)[0], -1])

    @staticmethod
    def from_numpy(arr: np.ndarray) -> tf.Tensor:
        "Convert a NumPy array to a tensor"
        # TODO change dtype
        return tf.constant(arr, dtype=tf.float32)

    @staticmethod
    def transpose(tensor: TensorType) -> tf.Tensor:
        "Transpose function for tensor of rank 2"
        return tf.transpose(tensor)

    @staticmethod
    def diag(tensor: TensorType) -> tf.Tensor:
        "Diagonal function: return the diagonal of a 2D tensor"
        return tf.linalg.diag_part(tensor)

    @staticmethod
    def reshape(tensor: TensorType, shape: List[int]) -> tf.Tensor:
        "Reshape function"
        return tf.reshape(tensor, shape)

    @staticmethod
    def equal(tensor: TensorType, other: Union[TensorType, int, float]) -> tf.Tensor:
        "Computes element-wise equality"
        return tf.math.equal(tensor, other)

    @staticmethod
    def pinv(tensor: TensorType) -> tf.Tensor:
        "Computes the pseudoinverse (Moore-Penrose inverse) of a matrix."
        return tf.linalg.pinv(tensor)

    @staticmethod
    def eigh(tensor: TensorType) -> tf.Tensor:
        "Computes the eigen decomposition of a self-adjoint matrix."
        eigval, eigvec = tf.linalg.eigh(tensor)
        return eigval, eigvec

    @staticmethod
    def quantile(tensor: TensorType, q: float, dim: int = None) -> tf.Tensor:
        "Computes the quantile of a tensor's components. q in (0,1)"
        q = tfp.stats.percentile(tensor, q * 100, axis=dim)
        return float(q) if dim is None else q

    @staticmethod
    def relu(tensor: TensorType) -> tf.Tensor:
        "Apply relu to a tensor"
        return tf.nn.relu(tensor)
