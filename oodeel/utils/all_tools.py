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
from typing import Any
from typing import Callable


def is_from(model_or_tensor: Any, framework: str) -> str:
    """Check wether a model or tensor belongs to a specific framework

    Args:
        model_or_tensor (Any): Neural network or Tensor
        framework (str):  Model or tensor framework ("torch" | "keras" | "tensorflow")

    Returns:
        bool: Wether the model belongs to specified framework or not
    """
    class_parents = list(
        map(
            lambda x: str(x).split("'")[1].split(".")[0],
            model_or_tensor.__class__.__mro__,
        )
    )
    return framework in class_parents


def softmax(tensor: Any):
    """Softmax function"""
    if is_from(tensor, "tensorflow"):
        from tensorflow.keras import activations

        return activations.softmax(tensor)
    elif is_from(tensor, "torch"):
        import torch.nn.functional as F

        return F.softmax(tensor)


def gradient_single(model: Callable, inputs: Any, targets: Any):
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
    if is_from(model, "keras"):
        from . import tf_tools

        grad_fn = tf_tools.gradient_single
    elif is_from(model, "torch"):
        from . import torch_tools

        grad_fn = torch_tools.gradient_single
    else:
        raise NotImplementedError()
    return grad_fn(model, inputs, targets)
