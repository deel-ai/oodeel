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
    matches one of the feature keys. Ne careful, the signature of decorated functions
    must be function(dataset, *args, **kwargs) with feature_key either in kwargs or
    args[0] when relevant.


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

        # If feature_key is provided, check that it is in the dataset feature keys
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
    """
    Class to manage tf.data.Dataset. The aim is to provide a simple interface for
    working with tf.data.Datasets and manage them without having to use
    tensorflow syntax.
    """

    def load_tf_ds_from_numpy(
        self, dataset_id: Union[np.ndarray, dict, tuple]
    ) -> tf.data.Dataset:
        """Load a tf.data.Dataset from a numpy array or a tuple/dict of numpy arrays

        Args:
            dataset_id (Union[np.ndarray, dict, tuple]): numpy array(s) to load.

        Returns:
            tf.data.Dataset
        """
        # If dataset_id is a numpy array, convert it to a dict
        if isinstance(dataset_id, np.ndarray):
            dataset_dict = {"input": dataset_id}

        # If dataset_id is a tuple, convert it to a dict
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
        """Load a tf.data.Dataset from a tf.data.Dataset by ensuring it has the
            correct format (dict-based)

        Args:
            dataset_id (tf.data.Dataset): tf.data.Dataset
            keys (list, optional): Keys to use for features if dataset_id is
                tuple based. Defaults to None.

        Returns:
            tf.data.Dataset
        """
        # If dataset_id is a tuple based tf.data.dataset, convert it to a dict
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
        """Load a tf.data.Dataset from the tensorflow_datasets catalog

        Args:
            dataset_id (str): Identifier of the dataset
            load_kwargs (dict, optional): Loading kwargs to add to tfds.load().
                Defaults to {}.

        Returns:
            tf.data.Dataset
        """
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
        """Turn a dict based tf.data.Dataset to a tuple based tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): Dict based tf.data.Dataset
            keys (list, optional): Features to use for the tuples based
                tf.data.Dataset. If None, takes all the features. Defaults to None.

        Returns:
            tf.data.Dataset
        """
        if keys is None:
            keys = list(dataset.element_spec.keys())
        dataset = dataset.map(lambda x: tuple(x[k] for k in keys))
        return dataset

    def tuple_to_dict(self, dataset: tf.data.Dataset, keys: list) -> tf.data.Dataset:
        """Turn a tuple based tf.data.Dataset to a dict based tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): Tuple based tf.data.Dataset
            keys (list): Keys to use for the dict based tf.data.Dataset

        Returns:
            tf.data.Dataset
        """
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
    ) -> tf.data.Dataset:
        """Assign a value to a feature for every samples in a tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to assigne the value to
            feature_key (str): Feature to assign the value to
            value (int): Value to assign

        Returns:
            tf.data.Dataset
        """
        assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"

        def assign_value_to_feature(x):
            x[feature_key] = value
            return x

        dataset = dataset.map(assign_value_to_feature)
        return dataset

    @dict_only_ds
    def get_feature_from_ds(
        self, dataset: tf.data.Dataset, feature_key: str
    ) -> np.ndarray:
        """Get a feature from a tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to get the feature from
            feature_key (str): Feature value to get

        Returns:
            np.ndarray: Feature values for dataset
        """
        features = dataset.map(lambda x: x[feature_key])
        features = list(features.as_numpy_iterator())
        features = np.array(features)
        return features

    @dict_only_ds
    def get_ds_feature_keys(self, dataset: tf.data.Dataset) -> list:
        """Get the feature keys of a tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to get the feature keys from

        Returns:
            list: List of feature keys
        """
        return list(dataset.element_spec.keys())

    def has_key(self, dataset: tf.data.Dataset, key: str) -> bool:
        """Check if a tf.data.Dataset has a feature denoted by key

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to check
            key (str): Key to check

        Returns:
            bool: If the tf.data.Dataset has a feature denoted by key
        """
        assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"
        return 1 if (key in dataset.element_spec.keys()) else 0

    def map_ds(
        self,
        dataset: tf.data.Dataset,
        map_fn: Callable,
        num_parallel_calls: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Map a function to a tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to map the function to
            map_fn (Callable): Function to map
            num_parallel_calls (Optional[int], optional): Number of parallel processes
                to use. Defaults to None.

        Returns:
            tf.data.Dataset: Maped dataset
        """
        dataset = dataset.map(map_fn, num_parallel_calls=num_parallel_calls)
        return dataset

    def prepare_for_training(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle_buffer_size: int = None,
        prefetch_buffer_size: Optional[int] = None,
        drop_remainder: Optional[bool] = False,
    ) -> tf.data.Dataset:
        """Prepare a tf.data.Dataset for training

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to prepare
            batch_size (int): Batch size
            shuffle_buffer_size (int): Shuffling buffer size. If None, shuffling is not
                performed. Defaults to None.
            prefetch_buffer_size (Optional[int], optional): Buffer size for prefetch.
                If None, automatically chose using tf.data.experimental.AUTOTUNE.
                Defaults to None.
            drop_remainder (Optional[bool], optional): To drop the last batch when
                its size is lower than batch_size. Defaults to False.

        Returns:
            tf.data.Dataset: Prepared dataset
        """
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if prefetch_buffer_size is not None:
            prefetch_buffer_size = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    def make_channel_first(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Make a tf.data.Dataset channel first. Make sur that the dataset is not
            already Channel first. If so, the tensor will have the format
            (batch_size, x_size, channel, y_size).

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to make channel first

        Returns:
            tf.data.Dataset: Channel first dataset
        """

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
                id_dataset input tensors. Defaults to None.
            channel_order (Optional[str], optional): channel order of the input

        Returns:
            tf.data.Dataset: merged dataset
        """
        len_elem_id = dataset_len_elem(id_dataset)
        len_elem_ood = dataset_len_elem(ood_dataset)
        assert (
            len_elem_id == len_elem_ood
        ), "incompatible dataset elements (different elem dict length)"

        # If a desired shape is given, triggers the resize
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

        # If the shape of the two datasets are different, triggers the resize
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
        """Filter a tf.data.Dataset by checking the value of a feature is in 'values'

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to filter
            feature_key (str): Feature name to check the value
            values (list): Feature_key values to keep (if excluded is False)
                or to exclude
            excluded (bool, optional): To keep (False) or exclude (True) the samples
                with Feature_key value included in Values. Defaults to False.

        Returns:
            tf.data.Dataset: Filtered dataset
        """
        # If the labels are one-hot encoded, prepare a function to get the label as int
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
