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
import torch

from ..types import Callable
from ..types import Union
from .operator import Operator


class TorchOperator(Operator):
    """Class to handle torch operations with a unified API"""

    @staticmethod
    def softmax(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Softmax function"""
        return torch.nn.functional.softmax(tensor)

    @staticmethod
    def argmax(
        tensor: Union[torch.Tensor, np.ndarray], axis: int = None
    ) -> torch.Tensor:
        """Argmax function"""
        return torch.argmax(tensor, dim=axis)

    @staticmethod
    def max(tensor: Union[torch.Tensor, np.ndarray], axis: int = None) -> torch.Tensor:
        """Max function"""
        return torch.max(tensor, dim=axis)

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
    def gradient(func: Callable, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute gradients for a batch of samples.
        Parameters

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
        return gradients
