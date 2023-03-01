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

from ..types import List
from ..types import Optional
from ..types import Tuple
from ..types import Union


def get_input_from_dataset_elem(elem):
    if isinstance(elem, (tuple, list)):
        tensor = elem[0]
    elif isinstance(elem, dict):
        tensor = elem[list(elem.keys())[0]]
    else:
        tensor = elem
    return tensor


def dataset_len_elem(dataset: tf.data.Dataset) -> int:
    if isinstance(dataset.element_spec, (tuple, list, dict)):
        return len(dataset.element_spec)
    return 1


def dataset_image_shape(dataset: tf.data.Dataset) -> Tuple[int]:
    """
    Get the shape of the images in the dataset

    Args:
        dataset:   input dataset
    Returns:
        shape of the images in the dataset
    """
    for x in dataset.take(1):
        if isinstance(x, tuple):
            shape = x[0].shape
        else:
            shape = x.shape
    return shape


def dataset_label_shape(dataset: tf.data.Dataset) -> Tuple[int]:
    for x in dataset.take(1):
        assert len(x) > 1, "No label to get the shape from"
        shape = x[1].shape
    return shape


def dataset_max_pixel(dataset: tf.data.Dataset) -> float:
    dataset = dataset_get_columns(dataset, 0)
    max_pixel = dataset.reduce(
        0.0, lambda x, y: float(tf.math.reduce_max(tf.maximum(x, y)))
    )
    return float(max_pixel)


def dataset_nb_labels(dataset: tf.data.Dataset) -> int:
    ds = dataset_get_columns(dataset, 1)
    ds = ds.unique()
    return len(list(ds.as_numpy_iterator()))


def dataset_cardinality(dataset: tf.data.Dataset) -> int:
    try:
        return len(dataset)
    except TypeError:
        cardinality = dataset.reduce(0, lambda x, _: x + 1)
        return int(cardinality)


def dataset_get_columns(
    dataset: tf.data.Dataset, columns: Union[int, List[int]]
) -> tf.data.Dataset:
    """
    Construct a dataset out of the columns of the input dataset.
    The columns are identified by "columns".
    Here columns means x, y, or ood_labels

    Args:
        dataset: input dataset
        columns: columns to extract

    Returns:
        tf.data.Dataset with columns extracted from the input dataset
    """
    if isinstance(columns, int):
        columns = [columns]
    length = dataset_len_elem(dataset)

    if length == 2:  # when image, label

        def return_columns_xy(x, y, col):
            X = [x, y]
            return tuple([X[i] for i in col])

        dataset = dataset.map(lambda x, y: return_columns_xy(x, y, columns))

    if length == 3:  # when image, label, ood_label or weights

        def return_columns_xyz(x, y, z, col):
            X = [x, y, z]
            return tuple([X[i] for i in col])

        dataset = dataset.map(lambda x, y, z: return_columns_xyz(x, y, z, columns))

    return dataset


def batch_tensor(
    tensors: Union[
        tf.data.Dataset, tf.Tensor, np.ndarray, Tuple[tf.Tensor], Tuple[np.ndarray]
    ],
    batch_size: int = 256,
    one_hot_encode: Optional[bool] = False,
    num_classes: Optional[int] = None,
):
    """
    Create a tensorflow dataset of tensors or series of tensors.

    Parameters
    ----------
    tensors
        Tuple of tensors or tensors to batch.
    batch_size
        Number of samples to iterate at once, if None process all at once.

    Returns
    -------
    dataset
        Tensorflow dataset batched.
    """
    if not isinstance(tensors, tf.data.Dataset):
        tensors = tf.data.Dataset.from_tensor_slices(tensors)

    if one_hot_encode:
        # check if it is one_hot_encoded
        label_shape = list(dataset_label_shape(tensors))
        if label_shape == []:
            nb_columns = dataset_len_elem(tensors)
            assert nb_columns == 2, "No labels to one-hot-encode"
            if num_classes is None:
                num_classes = dataset_nb_labels(tensors)
            dataset = tensors.map(lambda x, y: (x, tf.one_hot(y, num_classes))).batch(
                batch_size
            )
    else:
        dataset = tensors.batch(batch_size)

    return dataset
