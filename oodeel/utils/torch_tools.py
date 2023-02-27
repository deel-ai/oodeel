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


def softmax(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    return torch.nn.functional.softmax(tensor)


def argmax(tensor: Union[torch.Tensor, np.ndarray], axis: int = None) -> torch.Tensor:
    return torch.argmax(tensor, dim=axis)


def max(tensor: Union[torch.Tensor, np.ndarray], axis: int = None) -> torch.Tensor:
    return torch.max(tensor, dim=axis)


def one_hot(tensor: Union[torch.Tensor, np.ndarray], num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(tensor, num_classes)


def sign(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    return torch.sign(tensor)


def gradient_single(
    model: Callable, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradients for a batch of samples.
    Parameters
    ----------
    model
        Model used for computing gradient.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    Returns
    -------
    gradients
        Gradients computed, with the same shape as the inputs.
    """
    inputs.requires_grad_(True)
    score = torch.sum(model(inputs) * targets)
    gradients = torch.autograd.grad(score, inputs)
    inputs.requires_grad_(False)
    return gradients[0]
