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

from ..types import *


def keras_dataset_load(
    dataset_name: str, **kwargs
) -> Tuple[Tuple[Union[tf.data.Dataset, tf.Tensor, np.ndarray]]]:
    """
    Loads a dataset

    Parameters
    ----------
    dataset_name : str
    """
    assert hasattr(
        tf.keras.datasets, dataset_name
    ), f"{dataset_name} not available with keras.datasets"
    (x_train, y_train), (x_test, y_test) = getattr(
        tf.keras.datasets, dataset_name
    ).load_data(**kwargs)

    x_max = np.max(x_train)
    x_train = x_train.astype("float32") / x_max
    x_test = x_test.astype("float32") / x_max

    if dataset_name in ["mnist", "fashion_mnist"]:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    return (x_train, y_train), (x_test, y_test)
