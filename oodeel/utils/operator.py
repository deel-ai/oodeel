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


class Operator(ABC):
    @abstractmethod
    def softmax(tensor: Any) -> Any:
        """Softmax function"""
        raise NotImplementedError()

    @abstractmethod
    def argmax(tensor: Any, axis: int = None) -> Any:
        """Argmax function"""
        raise NotImplementedError()

    @abstractmethod
    def max(tensor: Any, axis: int = None) -> Any:
        """Max function"""
        raise NotImplementedError()

    @abstractmethod
    def one_hot(tensor: Any, num_classes: int) -> Any:
        """One hot function"""
        raise NotImplementedError()

    @abstractmethod
    def sign(tensor: Any) -> Any:
        """Sign function"""
        raise NotImplementedError()

    @abstractmethod
    def gradient(func: Callable, inputs: Any) -> Any:
        """
        Compute gradients for a batch of samples.
        Parameters
        ----------
        fun
            Function used for computing gradient.
        inputs
            Input samples to be explained.
        Returns
        -------
        gradients
            Gradients computed, with the same shape as the inputs.
        """
        raise NotImplementedError()
