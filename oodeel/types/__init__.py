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
"""
Typing module
"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np

avail_lib = []
try:
    import tensorflow as tf

    avail_lib.append("tensorflow")
except ImportError:
    pass

try:
    import torch

    avail_lib.append("torch")
except ImportError:
    pass


if len(avail_lib) == 2:
    DatasetType = Union[tf.data.Dataset, torch.utils.data.DataLoader]
    TensorType = Union[tf.Tensor, torch.Tensor, np.ndarray]
    ItemType = Union[
        tf.Tensor,
        torch.Tensor,
        np.ndarray,
        tuple,
        list,
        dict,
    ]

elif "tensorflow" in avail_lib:
    DatasetType = Type[tf.data.Dataset]
    TensorType = Union[tf.Tensor, np.ndarray]
    ItemType = Union[tf.Tensor, np.ndarray, tuple, list, dict]
elif "torch" in avail_lib:
    DatasetType = Type[torch.utils.data.DataLoader]
    TensorType = Union[torch.Tensor, np.ndarray]
    ItemType = Union[torch.Tensor, np.ndarray, tuple, list, dict]
