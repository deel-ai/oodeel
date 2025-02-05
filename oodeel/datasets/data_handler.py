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
import importlib
from abc import ABC
from abc import abstractmethod

import numpy as np

from ..types import Callable
from ..types import DatasetType
from ..types import ItemType
from ..types import Optional
from ..types import TensorType
from ..types import Tuple
from ..types import Union


def get_backend():
    """Detects whether TensorFlow or PyTorch is available and returns
    the preferred backend."""
    available_backends = []
    if importlib.util.find_spec("tensorflow"):
        available_backends.append("tensorflow")
    if importlib.util.find_spec("torch"):
        available_backends.append("torch")

    if len(available_backends) == 1:
        return available_backends[0]
    elif len(available_backends) == 0:
        raise ImportError("Neither TensorFlow nor PyTorch is installed.")
    else:
        raise ImportError(
            "Both TensorFlow and PyTorch are installed. Please specify the backend."
        )


def load_data_handler(backend: str = None):

    if backend is None:
        backend = get_backend()

    if backend == "tensorflow":
        from .tf_data_handler import TFDataHandler

        return TFDataHandler()

    elif backend == "torch":
        from .torch_data_handler import TorchDataHandler

        return TorchDataHandler()


class DataHandler(ABC):
    """
    Class to manage Datasets. The aim is to provide a simple interface
    for working with datasets (torch, tensorflow or other...) and manage them without
    having to use library-specific syntax.
    """

    def __init__(self):
        self.backend = None
        self.channel_order = None

    def split_by_class(
        self,
        dataset: DatasetType,
        in_labels: Optional[Union[np.ndarray, list]] = None,
        out_labels: Optional[Union[np.ndarray, list]] = None,
    ) -> Optional[Tuple[DatasetType]]:
        """Filter the dataset by assigning ood labels depending on labels
        value (typically, class id).

        Args:
            in_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as in-distribution. Defaults to None.
            out_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as out-of-distribution. Defaults to None.

        Returns:
            Optional[Tuple[OODDataset]]: Tuple of in-distribution and
                out-of-distribution OODDatasets
        """
        # Make sure the dataset has labels
        assert (in_labels is not None) or (
            out_labels is not None
        ), "specify labels to filter with"
        assert self.get_item_length(dataset) >= 2, "the dataset has no labels"

        # Filter the dataset depending on in_labels and out_labels given
        if (out_labels is not None) and (in_labels is not None):
            in_data = self.filter_by_feature_value(dataset, "label", in_labels)
            out_data = self.filter_by_feature_value(dataset, "label", out_labels)

        if out_labels is None:
            in_data = self.filter_by_feature_value(dataset, "label", in_labels)
            out_data = self.filter_by_feature_value(
                dataset, "label", in_labels, excluded=True
            )

        elif in_labels is None:
            in_data = self.filter_by_feature_value(
                dataset, "label", out_labels, excluded=True
            )
            out_data = self.filter_by_feature_value(dataset, "label", out_labels)

        # Return the filtered OODDatasets
        return in_data, out_data

    @classmethod
    @abstractmethod
    def prepare(
        cls,
        dataset: DatasetType,
        batch_size: int,
        preprocess_fn: Optional[Callable] = None,
        augment_fn: Optional[Callable] = None,
        columns: Optional[list] = None,
        shuffle: bool = False,
        dict_based_fns: bool = True,
        **kwargs_prepare,
    ) -> DatasetType:
        """Prepare dataset for scoring or training

        Args:
            batch_size (int): Batch size
            preprocess_fn (Callable, optional): Preprocessing function to apply to
                the dataset. Defaults to None.
            augment_fn (Callable, optional): Augment function to be used (when the
                returned dataset is to be used for training). Defaults to None.
            columns (list, optional): List of keys corresponding to the features
                that will be returned. Keep all features if None. Defaults to None.
            shuffle (bool, optional): To shuffle the returned dataset or not.
                Defaults to False.
            dict_based_fns (bool): Whether to use preprocess and DA functions as dict
                based (if True) or as tuple based (if False). Defaults to True.
            kwargs_prepare (dict): Additional parameters to be passed to the
                data_handler for backend specific preparation.


        Returns:
            DatasetType: prepared dataset
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_dataset_from_arrays(
        dataset_id: ItemType, keys: Optional[list] = None
    ) -> DatasetType:
        """Load a tf.data.Dataset from a np.ndarray, a tf.Tensor or a tuple/dict
        of np.ndarrays/DatasetType.

        Args:
            dataset_id (ItemType): numpy array(s) to load.
            keys (list, optional): Features keys. If None, assigned as "input_i"
                for i-th feature. Defaults to None.

        Returns:
            DatasetType
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_custom_dataset(
        dataset_id: DatasetType, keys: Optional[list] = None
    ) -> DatasetType:
        """Load a custom dataset by ensuring it is properly formatted.

        Args:
            dataset_id: dataset
            keys: feature keys

        Returns:
            A properly formatted dataset.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_ds_feature_keys(dataset: DatasetType) -> list:
        """Get the feature keys of a Dataset

        Args:
            dataset (Dataset): Dataset to get the feature keys from

        Returns:
            list: List of feature keys
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def map_ds(dataset: DatasetType, map_fn: Callable) -> DatasetType:
        """Map a function to a Dataset

        Args:
            dataset (DatasetType): Dataset to map the function to
            map_fn (Callable): Function to map

        Returns:
            DatasetType: Mapped dataset
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def filter_by_feature_value(
        dataset: DatasetType,
        feature_key: str,
        values: list,
        excluded: bool = False,
    ) -> DatasetType:
        """Filter the dataset by checking the value of a feature is in `values`

        Args:
            dataset (Dataset): Dataset to filter
            feature_key (str): Feature name to check the value
            values (list): Feature_key values to keep (if excluded is False)
                or to exclude
            excluded (bool, optional): To keep (False) or exclude (True) the samples
                with Feature_key value included in Values. Defaults to False.

        Returns:
            DatasetType: Filtered dataset
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_item_length(dataset: DatasetType) -> int:
        """Number of elements in a dataset item

        Args:
            dataset (DatasetType): Dataset

        Returns:
            int: Item length
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_dataset_length(dataset: DatasetType) -> int:
        """Number of items in a dataset

        Args:
            dataset (DatasetType): Dataset

        Returns:
            int: Dataset length
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_feature_shape(dataset: DatasetType, feature_key: Union[str, int]) -> tuple:
        """Get the shape of a feature of dataset identified by feature_key

        Args:
            dataset (Dataset): a Dataset
            feature_key (Union[str, int]): The identifier of the feature

        Returns:
            tuple: the shape of feature_id
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_input_from_dataset_item(elem: ItemType) -> TensorType:
        """Get the tensor that is to be feed as input to a model from a dataset element.

        Args:
            elem (ItemType): dataset element to extract input from

        Returns:
            TensorType: Input tensor
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_label_from_dataset_item(item: ItemType):
        """Retrieve label tensor from item as a tuple/list. Label must be at index 1
        in the item tuple. If one-hot encoded, labels are converted to single value.

        Args:
            elem (ItemType): dataset element to extract label from

        Returns:
            Any: Label tensor
        """
        raise NotImplementedError()
