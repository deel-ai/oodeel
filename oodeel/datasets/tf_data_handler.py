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
from typing import get_args

import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset as hf_load_dataset

from ..types import Callable
from ..types import ItemType
from ..types import Optional
from ..types import TensorType
from ..types import Tuple
from ..types import Union
from .data_handler import DataHandler


def dict_only_ds(ds_handling_method: Callable) -> Callable:
    """Decorator to ensure that the dataset is a dict dataset and that the column_name
    given as argument matches one of the column names.
    matches one of the column names. The signature of decorated functions
    must be function(dataset, *args, **kwargs) with column_name either in kwargs or
    args[0] when relevant.


    Args:
        ds_handling_method: method to decorate

    Returns:
        decorated method
    """

    def wrapper(dataset: tf.data.Dataset, *args, **kwargs):
        assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"

        if "column_name" in kwargs.keys():
            column_name = kwargs["column_name"]
        elif len(args) > 0:
            column_name = args[0]

        # If column_name is provided, check that it is in the dataset column names
        if (len(args) > 0) or ("column_name" in kwargs):
            if isinstance(column_name, str):
                column_name = [column_name]
            for name in column_name:
                assert (
                    name in dataset.element_spec.keys()
                ), f"The input dataset has no column named {name}"
        return ds_handling_method(dataset, *args, **kwargs)

    return wrapper


class TFDataHandler(DataHandler):
    """
    Class to manage tf.data.Dataset. The aim is to provide a simple interface for
    working with tf.data.Datasets and manage them without having to use
    tensorflow syntax.
    """

    def __init__(self) -> None:
        """
        Initializes the TFDataHandler instance.
        Attributes:
            backend (str): The backend framework used, set to "tensorflow".
            channel_order (str): The channel order format, set to "channels_last".
        """
        super().__init__()
        self.backend = "tensorflow"
        self.channel_order = "channels_last"

    @classmethod
    def load_dataset(
        cls,
        dataset_id: Union[tf.data.Dataset, ItemType, str],
        columns: Optional[list] = None,
        hub: Optional[str] = "tensorflow-datasets",
        load_kwargs: dict = {},
    ) -> tf.data.Dataset:
        """Load dataset from different manners, ensuring to return a dict based
        tf.data.Dataset.

        Args:
            dataset_id (Union[tf.data.Dataset, ItemType, str]): dataset identification.
            Can be the name of a dataset from tensorflow_datasets, a tf.data.Dataset,
            or a tuple/dict of np.ndarrays/tf.Tensors.
            columns (list, optional): Column names. If None, assigned as "input_i"
                for i-th column. Defaults to None.
            load_kwargs (dict, optional): Additional args for loading from
                tensorflow_datasets. Defaults to {}.

        Returns:
            tf.data.Dataset: A dict based tf.data.Dataset
        """

        assert hub in {
            "tensorflow-datasets",
            "huggingface",
        }, "hub must be either 'tensorflow-datasets' or 'huggingface'"

        if isinstance(dataset_id, get_args(ItemType)):
            dataset = cls.load_dataset_from_arrays(dataset_id, columns)
        elif isinstance(dataset_id, tf.data.Dataset):
            dataset = cls.load_custom_dataset(dataset_id, columns)
        elif isinstance(dataset_id, str):
            if hub == "tensorflow-datasets":
                load_kwargs["as_supervised"] = False
                dataset = cls.load_from_tensorflow_datasets(dataset_id, load_kwargs)
            elif hub == "huggingface":
                dataset = cls.load_from_huggingface(dataset_id, load_kwargs)
        return dataset

    @staticmethod
    def load_dataset_from_arrays(
        dataset_id: ItemType, columns: Optional[list] = None
    ) -> tf.data.Dataset:
        """Load a tf.data.Dataset from a np.ndarray, a tf.Tensor or a tuple/dict
        of np.ndarrays/tf.Tensors.

        Args:
            dataset_id (ItemType): numpy array(s) to load.
            columns (list, optional): Column names to assign. If None,
                assigned as "input_i" for i-th column. Defaults to None.

        Returns:
            tf.data.Dataset
        """
        # If dataset_id is a numpy array, convert it to a dict
        if isinstance(dataset_id, get_args(TensorType)):
            dataset_dict = {"input": dataset_id}

        # If dataset_id is a tuple, convert it to a dict
        elif isinstance(dataset_id, tuple):
            len_elem = len(dataset_id)
            if columns is None:
                if len_elem == 2:
                    dataset_dict = {"input": dataset_id[0], "label": dataset_id[1]}
                else:
                    dataset_dict = {
                        f"input_{i}": dataset_id[i] for i in range(len_elem - 1)
                    }
                    dataset_dict["label"] = dataset_id[-1]
                print(
                    'Loading tf.data.Dataset with elems as dicts, assigning "input_i" '
                    'key to the i-th tuple dimension and "label" key to the last '
                    "tuple dimension."
                )
            else:
                assert (
                    len(columns) == len_elem
                ), "Number of column names mismatch with the number of columns"
                dataset_dict = {columns[i]: dataset_id[i] for i in range(len_elem)}

        elif isinstance(dataset_id, dict):
            if columns is not None:
                len_elem = len(dataset_id)
                assert (
                    len(columns) == len_elem
                ), "Number of column names mismatch with the number of columns"
                original_columns = list(dataset_id.keys())
                dataset_dict = {
                    columns[i]: dataset_id[original_columns[i]] for i in range(len_elem)
                }

        dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
        return dataset

    @classmethod
    def load_custom_dataset(
        cls, dataset_id: tf.data.Dataset, columns: Optional[list] = None
    ) -> tf.data.Dataset:
        """Load a custom Dataset by ensuring it has the correct format (dict-based)

        Args:
            dataset_id (tf.data.Dataset): tf.data.Dataset
            columns (list, optional): Column names to use for elements if dataset_id is
                tuple based. If None, assigned as "input_i"
                for i-th column. Defaults to None.

        Returns:
            tf.data.Dataset
        """
        # If dataset_id is a tuple based tf.data.dataset, convert it to a dict
        if not isinstance(dataset_id.element_spec, dict):
            len_elem = len(dataset_id.element_spec)
            if columns is None:
                print(
                    "Column name not found, assigning 'input_i' "
                    "key to the i-th tensor and 'label' key to the last"
                )
                if len_elem == 2:
                    columns = ["input", "label"]
                else:
                    columns = [f"input_{i}" for i in range(len_elem)]
                    columns[-1] = "label"
            else:
                assert (
                    len(columns) == len_elem
                ), "Number of column names mismatch with the number of columns"

            dataset_id = cls.tuple_to_dict(dataset_id, columns)

        dataset = dataset_id
        return dataset

    @staticmethod
    def load_from_huggingface(
        dataset_id: str,
        load_kwargs: dict = {},
    ) -> tf.data.Dataset:
        """Load a Dataset from the Hugging Face datasets catalog

        Args:
            dataset_id (str): Identifier of the dataset
            load_kwargs (dict): Loading kwargs to add to the initialization
            of the dataset.

        Returns:
            tf.data.Dataset: dataset
        """
        dataset = hf_load_dataset(dataset_id, **load_kwargs)
        dataset = dataset.to_tf_dataset()
        return dataset

    @staticmethod
    def load_from_tensorflow_datasets(
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
        dataset = tfds.load(dataset_id, **load_kwargs)
        return dataset

    @staticmethod
    @dict_only_ds
    def dict_to_tuple(
        dataset: tf.data.Dataset, columns: Optional[list] = None
    ) -> tf.data.Dataset:
        """Turn a dict based tf.data.Dataset to a tuple based tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): Dict based tf.data.Dataset
            columns (list, optional): Columns to use for the tuples based
                tf.data.Dataset. If None, takes all the columns. Defaults to None.

        Returns:
            tf.data.Dataset
        """
        if columns is None:
            columns = list(dataset.element_spec.keys())
        dataset = dataset.map(lambda x: tuple(x[k] for k in columns))
        return dataset

    @staticmethod
    def tuple_to_dict(dataset: tf.data.Dataset, columns: list) -> tf.data.Dataset:
        """Turn a tuple based tf.data.Dataset to a dict based tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): Tuple based tf.data.Dataset
            columns (list): Column names to use for the dict based tf.data.Dataset

        Returns:
            tf.data.Dataset
        """
        assert isinstance(
            dataset.element_spec, tuple
        ), "dataset elements must be tuples"
        len_elem = len(dataset.element_spec)
        assert len_elem == len(
            columns
        ), "The number of columns must be equal to the number of tuple elements"

        def tuple_to_dict(*inputs):
            return {columns[i]: inputs[i] for i in range(len_elem)}

        dataset = dataset.map(tuple_to_dict)
        return dataset

    @staticmethod
    @dict_only_ds
    def get_ds_column_names(dataset: tf.data.Dataset) -> list:
        """Get the column names of a tf.data.Dataset

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to get the column names from

        Returns:
            list: List of column names
        """
        return list(dataset.element_spec.keys())

    @staticmethod
    def map_ds(
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
        if num_parallel_calls is None:
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        dataset = dataset.map(map_fn, num_parallel_calls=num_parallel_calls)
        return dataset

    @staticmethod
    @dict_only_ds
    def filter_by_value(
        dataset: tf.data.Dataset,
        column_name: str,
        values: list,
        excluded: bool = False,
    ) -> tf.data.Dataset:
        """Filter a tf.data.Dataset by checking if the value of a column is in 'values'

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to filter
            column_name (str): Column to filter the dataset with
            values (list): Column values to keep (if excluded is False)
                or to exclude
            excluded (bool, optional): To keep (False) or exclude (True) the samples
                with Column values included in Values. Defaults to False.

        Returns:
            tf.data.Dataset: Filtered dataset
        """
        # If the labels are one-hot encoded, prepare a function to get the label as int
        if len(dataset.element_spec[column_name].shape) > 0:

            def get_label_int(elem):
                return int(tf.argmax(elem[column_name]))

        else:

            def get_label_int(elem):
                return elem[column_name]

        def filter_fn(elem):
            value = get_label_int(elem)
            if excluded:
                return not tf.reduce_any(tf.equal(value, values))
            else:
                return tf.reduce_any(tf.equal(value, values))

        dataset_to_filter = dataset
        dataset_to_filter = dataset_to_filter.filter(filter_fn)
        return dataset_to_filter

    @classmethod
    def prepare(
        cls,
        dataset: tf.data.Dataset,
        batch_size: int,
        preprocess_fn: Optional[Callable] = None,
        augment_fn: Optional[Callable] = None,
        columns: Optional[list] = None,
        shuffle: bool = False,
        dict_based_fns: bool = True,
        return_tuple: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        prefetch_buffer_size: Optional[int] = None,
        drop_remainder: Optional[bool] = False,
    ) -> tf.data.Dataset:
        """Prepare a tf.data.Dataset for training

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to prepare
            batch_size (int): Batch size
            preprocess_fn (Callable, optional): Preprocessing function to apply to
                the dataset. Defaults to None.
            augment_fn (Callable, optional): Augment function to be used (when the
                returned dataset is to be used for training). Defaults to None.
            columns (list, optional): List of column names corresponding to the columns
                that will be returned. Keep all columns if None. Defaults to None.
            shuffle (bool, optional): To shuffle the returned dataset or not.
                Defaults to False.
            dict_based_fns (bool): Whether to use preprocess and DA functions as dict
                based (if True) or as tuple based (if False). Defaults to True.
            return_tuple (bool, optional): Whether to return each dataset item
                as a tuple. Defaults to True.
            shuffle_buffer_size (int, optional): Size of the shuffle buffer. If None,
                taken as the number of samples in the dataset. Defaults to None.
            prefetch_buffer_size (Optional[int], optional): Buffer size for prefetch.
                If None, automatically chose using tf.data.experimental.AUTOTUNE.
                Defaults to None.
            drop_remainder (Optional[bool], optional): To drop the last batch when
                its size is lower than batch_size. Defaults to False.

        Returns:
            tf.data.Dataset: Prepared dataset
        """
        # dict based to tuple based
        columns = columns or cls.get_ds_column_names(dataset)
        if not dict_based_fns:
            dataset = cls.dict_to_tuple(dataset, columns)

        # preprocess + DA
        if preprocess_fn is not None:
            dataset = cls.map_ds(dataset, preprocess_fn)
        if augment_fn is not None:
            dataset = cls.map_ds(dataset, augment_fn)

        if dict_based_fns and return_tuple:
            dataset = cls.dict_to_tuple(dataset, columns)

        dataset = dataset.cache()

        # shuffle
        if shuffle:
            num_samples = cls.get_dataset_length(dataset)
            shuffle_buffer_size = (
                num_samples if shuffle_buffer_size is None else shuffle_buffer_size
            )
            dataset = dataset.shuffle(shuffle_buffer_size)
        # batch
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        # prefetch
        if prefetch_buffer_size is not None:
            prefetch_buffer_size = tf.data.experimental.AUTOTUNE
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    @staticmethod
    def make_channel_first(input_key: str, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Make a tf.data.Dataset channel first. Make sure that the dataset is not
            already Channel first. If so, the tensor will have the format
            (batch_size, x_size, channel, y_size).

        Args:
            input_key (str): input key of the dict-based tf.data.Dataset
            dataset (tf.data.Dataset): tf.data.Dataset to make channel first

        Returns:
            tf.data.Dataset: Channel first dataset
        """

        def channel_first(x):
            x[input_key] = tf.transpose(x[input_key], perm=[2, 0, 1])
            return x

        dataset = dataset.map(channel_first)
        return dataset

    @staticmethod
    def get_item_length(dataset: tf.data.Dataset) -> int:
        """Get the length of a dataset element. If an element is a tensor, the length is
        one and if it is a sequence (list or tuple), it is len(elem).

        Args:
            dataset (tf.data.Dataset): Dataset to process

        Returns:
            int: length of the dataset elems
        """
        if isinstance(dataset.element_spec, (tuple, list, dict)):
            return len(dataset.element_spec)
        return 1

    @staticmethod
    def get_dataset_length(dataset: tf.data.Dataset) -> int:
        """Get the length of a dataset. Try to access it with len(), and if not
        available, with a reduce op.

        Args:
            dataset (tf.data.Dataset): Dataset to process

        Returns:
            int: _description_
        """
        try:
            return len(dataset)
        except TypeError:
            cardinality = dataset.reduce(0, lambda x, _: x + 1)
            return int(cardinality)

    @staticmethod
    def get_column_elements_shape(
        dataset: tf.data.Dataset, column_name: Union[str, int]
    ) -> tuple:
        """Get the shape of the elements of a column of dataset identified by
        column_name

        Args:
            dataset (tf.data.Dataset): a tf.data.dataset
            column_name (Union[str, int]): The column name to get
                the element shape from.

        Returns:
            tuple: the shape of an element from column_name
        """
        return tuple(dataset.element_spec[column_name].shape)

    @staticmethod
    def get_columns_shapes(dataset: tf.data.Dataset) -> dict:
        """Get the shapes of the elements of all columns of a dataset

        Args:
            dataset (Dataset): a Dataset

        Returns:
            dict: dictionary of column names and their corresponding shape
        """

        if isinstance(dataset.element_spec, tuple):
            shapes = [None for _ in range(len(dataset.element_spec))]
            for i in range(len(dataset.element_spec)):
                try:
                    shapes[i] = tuple(dataset.element_spec[i].shape)
                except AttributeError:
                    pass
            shapes = tuple(shapes)
        elif isinstance(dataset.element_spec, dict):
            shapes = {}
            for key in dataset.element_spec.keys():
                try:
                    shapes[key] = tuple(dataset.element_spec[key].shape)
                except AttributeError:
                    pass
        return shapes

    @staticmethod
    def get_input_from_dataset_item(elem: ItemType) -> TensorType:
        """Get the tensor that is to be feed as input to a model from a dataset element.

        Args:
            elem (ItemType): dataset element to extract input from

        Returns:
            TensorType: Input tensor
        """
        if isinstance(elem, (tuple, list)):
            tensor = elem[0]
        elif isinstance(elem, dict):
            tensor = elem[list(elem.keys())[0]]
        else:
            tensor = elem
        return tensor

    @staticmethod
    def get_label_from_dataset_item(item: ItemType):
        """Retrieve label tensor from item as a tuple/list. Label must be at index 1
        in the item tuple. If one-hot encoded, labels are converted to single value.

        Args:
            elem (ItemType): dataset element to extract label from

        Returns:
            Any: Label tensor
        """
        label = item[1]  # labels must be at index 1 in the item tuple
        # If labels are one-hot encoded, take the argmax
        if tf.rank(label) > 1 and label.shape[1] > 1:
            label = tf.reshape(label, shape=[label.shape[0], -1])
            label = tf.argmax(label, axis=1)
        # If labels are in two dimensions, squeeze them
        if len(label.shape) > 1:
            label = tf.reshape(label, [label.shape[0]])
        return label
