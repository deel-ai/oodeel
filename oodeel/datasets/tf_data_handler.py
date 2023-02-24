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
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class TFDataHandler(object):
    def __init__(self, with_gpu: bool = True):
        self.with_gpu = with_gpu
        if not with_gpu:
            tf.config.set_visible_devices([], "GPU")

    def load_tf_ds_from_numpy(
        self, dataset_id: Union[np.ndarray, dict, tuple]
    ) -> tf.data.Dataset:
        if isinstance(dataset_id, np.ndarray):
            dataset_dict = {"input": dataset_id}

        elif isinstance(dataset_id, tuple):
            len_elem = len(dataset_id)
            if len_elem == 2:
                dataset_dict = {"input": dataset_id[0], "label": dataset_id[1]}
            else:
                dataset_dict = {
                    f"input_{i}": dataset_id[i] for i in range(len_elem - 1)
                }
                dataset_dict["label"] = dataset_id[-1]
            print(
                'Loading tf.data.Dataset with elems as dicts, assigning "input_i" key'
                ' to the i-th tuple dimension and "label" key to the last '
                "tuple dimension."
            )
            dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
            return dataset

        elif isinstance(dataset_id, dict):
            dataset = tf.data.Dataset.from_tensor_slices(dataset_id)

    def load_tf_ds(
        self, dataset_id: tf.data.Dataset, keys: list = None
    ) -> tf.data.Dataset:
        if not isinstance(dataset_id.element_spec, dict):
            print(
                "Feature name not found, assigning 'input_i' "
                "key to the i-th tensor and 'label' key to the last"
            )
            len_elem = len(dataset_id.element_spec)

            if keys is None:
                if len_elem == 2:
                    keys = ["input", "label"]
                else:
                    keys = [f"input_{i}" for i in range(len_elem)]
                    keys[-1] = "label"

            def tuple_to_dict(*inputs):
                return {keys[i]: inputs[i] for i in range(len_elem)}

            dataset_id = dataset_id.map(tuple_to_dict)

        dataset = dataset_id
        return dataset

    def load_tf_ds_from_tfds(
        self,
        dataset_id: str,
        load_kwargs: dict = {},
    ) -> tf.data.Dataset:
        assert (
            dataset_id in tfds.list_builders()
        ), "Dataset not available on tensorflow datasets catalog"
        load_kwargs["with_info"] = True
        dataset, infos = tfds.load(dataset_id, **load_kwargs)
        return dataset, infos


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
