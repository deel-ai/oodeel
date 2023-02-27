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

from ..types import Callable
from ..types import Union
from .operator import Operator


class TFOperator(Operator):
    def softmax(tensor: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """Softmax function"""
        return tf.keras.activations.softmax(tensor)

    def argmax(tensor: Union[tf.Tensor, np.ndarray], axis: int = None) -> tf.Tensor:
        """Argmax function"""
        return tf.argmax(tensor, axis=axis)

    def max(tensor: Union[tf.Tensor, np.ndarray], axis: int = None) -> tf.Tensor:
        """Max function"""
        return tf.reduce_max(tensor, axis=axis)

    def one_hot(tensor: Union[tf.Tensor, np.ndarray], num_classes: int) -> tf.Tensor:
        """One hot function"""
        return tf.one_hot(tensor, num_classes)

    def sign(tensor: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """Sign function"""
        return tf.sign(tensor)

    @tf.function
    def gradient_model(
        model: Callable, inputs: tf.Tensor, targets: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradients of the output dimension corresponding to the prediction
        of a model w.r.t inputs for a batch of samples.
        ----------
        model
            Model used for computing gradient.
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}),
            one for each sample.
        Returns
        -------
        gradients
            Gradients computed, with the same shape as the inputs.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            score = tf.reduce_sum(tf.multiply(model(inputs), targets), axis=1)
        return tape.gradient(score, inputs)

    @tf.function
    def gradient(func: Callable, inputs: tf.Tensor) -> tf.Tensor:
        """
        Compute gradients for a batch of samples.
        Parameters
        ----------
        fun
            Function used for computing gradient. Must be built with tf differentiable
            operations only
        inputs
            Input samples to be explained.
        Returns
        -------
        gradients
            Gradients computed, with the same shape as the inputs.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            outputs = func(inputs)
        raise tape.gradient(outputs, inputs)
