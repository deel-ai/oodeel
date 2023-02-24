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
import tensorflow_datasets as tfds

from ..types import Callable
from ..types import Optional
from ..types import Tuple
from ..types import Union
from ..utils import dataset_len_elem


def dict_only_ds(ds_handling_method: Callable) -> Callable:
    """Decorator to ensure that the dataset is a dict dataset and that the input key
    matches one of the feature keys

    Args:
        ds_handling_method: method to decorate

    Returns:
        decorated method
    """

    def wrapper(self, dataset: tf.data.Dataset, *args, **kwargs):
        assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"

        if "feature_key" in kwargs:
            feature_key = kwargs["feature_key"]
        elif len(args) > 0:
            feature_key = args[0]

        if (len(args) > 0) or ("feature_key" in kwargs):
            if isinstance(feature_key, str):
                feature_key = [feature_key]
            for key in feature_key:
                assert (
                    key in dataset.element_spec.keys()
                ), f"The input dataset has no feature names {key}"
        return ds_handling_method(self, dataset, *args, **kwargs)

    return wrapper


class TFDataHandler(object):
    # TODO only static methods ?
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
            if keys is None:
                len_elem = len(dataset_id.element_spec)
                if len_elem == 2:
                    keys = ["input", "label"]
                else:
                    keys = [f"input_{i}" for i in range(len_elem)]
                    keys[-1] = "label"

            dataset_id = self.tuple_to_dict(dataset_id, keys)

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

    @dict_only_ds
    def dict_to_tuple(
        self, dataset: tf.data.Dataset, keys: list = None
    ) -> tf.data.Dataset:
        if keys is None:
            keys = list(dataset.element_spec.keys())
        dataset = dataset.map(lambda x: tuple(x[k] for k in keys))
        return dataset

    def tuple_to_dict(self, dataset: tf.data.Dataset, keys: list) -> tf.data.Dataset:
        assert isinstance(
            dataset.element_spec, tuple
        ), "dataset elements must be tuples"
        len_elem = len(dataset.element_spec)
        assert len_elem == len(
            keys
        ), "The number of keys must be equal to the number of tuple elements"

        def tuple_to_dict(*inputs):
            return {keys[i]: inputs[i] for i in range(len_elem)}

        dataset = dataset.map(tuple_to_dict)
        return dataset

    def assign_feature_value(
        self, dataset: tf.data.Dataset, feature_key: str, value: int
    ):
        assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"

        def assign_value_to_feature(x):
            x[feature_key] = value
            return x

        dataset = dataset.map(assign_value_to_feature)
        return dataset

    @dict_only_ds
    def get_feature_from_ds(self, dataset: tf.data.Dataset, feature_key: str) -> dict:
        features = dataset.map(lambda x: x[feature_key])
        features = list(features.as_numpy_iterator())
        features = np.array(features)
        return features

    @dict_only_ds
    def get_ds_feature_keys(self, dataset: tf.data.Dataset) -> list:
        return list(dataset.element_spec.keys())

    def has_key(self, dataset: tf.data.Dataset, key: str) -> list:
        assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"
        return 1 if (key in dataset.element_spec.keys()) else 0

    @dict_only_ds
    def filter_ds_by_feature_value(
        dataset: tf.data.Dataset, feature_key: str, value: list
    ):
        def filter_by_value(x):
            return x[feature_key] == value

        dataset = dataset.filter(filter_by_value)
        return dataset

    def map_ds(
        self,
        dataset: tf.data.Dataset,
        map_fn: Callable,
        num_parallel_calls: Optional[int] = None,
    ) -> tf.data.Dataset:
        dataset = dataset.map(map_fn, num_parallel_calls=num_parallel_calls)
        return dataset

    def prepare_for_training(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle_buffer_size: int,
        prefetch_buffer_size: Optional[int] = None,
        drop_remainder: Optional[bool] = False,
    ) -> tf.data.Dataset:
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if prefetch_buffer_size is not None:
            prefetch_buffer_size = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    def make_channel_first(dataset: tf.data.Dataset) -> tf.data.Dataset:
        def channel_first(x):
            return tf.transpose(x, perm=[2, 0, 1])

        dataset = dataset.map(channel_first)
        return dataset

    def merge(
        self,
        id_dataset: tf.data.Dataset,
        ood_dataset: tf.data.Dataset,
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
        channel_order: Optional[str] = "channels_last",
    ) -> tf.data.Dataset:
        """Merge two tf.data.Datasets

        Args:
            id_dataset (tf.data.Dataset): dataset of in-distribution data
            ood_dataset (tf.data.Dataset): dataset of out-of-distribution data
                resize (Optional[bool], optional): toggles if input tensors of the
                datasets have to be resized to have the same shape. Defaults to True.
            shape (Optional[Tuple[int]], optional): shape to use for resizing input
                tensors. If None, the tensors are resized with the shape of the
                id_dataset
            input tensors . Defaults to None.

        Returns:
            tf.data.Dataset: merged dataset with ood labels
        """
        len_elem_id = dataset_len_elem(id_dataset)
        len_elem_ood = dataset_len_elem(ood_dataset)
        assert (
            len_elem_id == len_elem_ood
        ), "incompatible dataset elements (different elem dict length)"

        if shape is not None:
            resize = True

        id_elem_spec = id_dataset.element_spec
        ood_elem_spec = ood_dataset.element_spec
        assert isinstance(id_elem_spec, dict), "dataset elements must be dicts"
        assert isinstance(ood_elem_spec, dict), "dataset elements must be dicts"

        input_key_id = list(id_elem_spec.keys())[0]
        input_key_ood = list(ood_elem_spec.keys())[0]
        shape_id = id_dataset.element_spec[input_key_id].shape
        shape_ood = ood_dataset.element_spec[input_key_ood].shape

        if shape_id != shape_ood:
            resize = True

            if shape is None:
                print(
                    "Resizing the first item of elem (usually the image)",
                    " with the shape of id_dataset",
                )
                if channel_order == "channels_first":
                    shape = shape_id[1:]
                else:
                    shape = shape_id[:2]

        if resize:

            def reshape_im_id(elem):
                elem[input_key_id] = tf.image.resize(elem[input_key_id], shape)
                return elem

            def reshape_im_ood(elem):
                elem[input_key_ood] = tf.image.resize(elem[input_key_ood], shape)
                return elem

            id_dataset = id_dataset.map(reshape_im_id)
            ood_dataset = ood_dataset.map(reshape_im_ood)

        merged_dataset = id_dataset.concatenate(ood_dataset)
        return merged_dataset

    @dict_only_ds
    def filter_by_feature_value(
        self,
        dataset: tf.data.Dataset,
        feature_key: str,
        values: list,
        excluded: bool = False,
    ) -> tf.data.Dataset:
        if len(dataset.element_spec[feature_key].shape) > 0:

            def get_label_int(elem):
                return int(tf.argmax(elem[feature_key]))

        else:

            def get_label_int(elem):
                return elem[feature_key]

        def filter_fn(elem):
            value = get_label_int(elem)
            if excluded:
                return not tf.reduce_any(tf.equal(value, values))
            else:
                return tf.reduce_any(tf.equal(value, values))

        dataset_to_filter = dataset
        dataset_to_filter = dataset_to_filter.filter(filter_fn)
        return dataset_to_filter


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
