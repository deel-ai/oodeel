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
import torch

from typing import Callable


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


if __name__ == '__main__':
    import torch.nn as nn
    import torch.nn.functional as F

    inputs = torch.randn(16, 3, 32, 32)  # bs, c, h, w
    targets = F.one_hot(torch.randint(10, (16,)))
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, 1, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*32*16, 10)
    )
    gradients = gradient_single(model, inputs, targets)
    print(gradients.shape, inputs.requires_grad)
