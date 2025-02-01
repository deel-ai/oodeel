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
from abc import ABC
from abc import abstractmethod

import numpy as np

from ..types import Callable
from ..types import DatasetType
from ..types import ItemType
from ..types import Optional
from ..types import Tuple
from ..types import Union


class DataHandler(ABC):
    """
    Class to manage Datasets. The aim is to provide a simple interface
    for working with datasets (torch, tensorflow or other...) and manage them without
    having to use library-specific syntax.
    """

    @classmethod
    @abstractmethod
    def load_dataset(
        cls,
        dataset_id: Union[ItemType, DatasetType, str],
        keys: Optional[list] = None,
        load_kwargs: dict = {},
    ) -> DatasetType:
        """Load dataset from different manners

        Args:
            dataset_id (Union[ItemType, DatasetType, str]): dataset identification
            keys (list, optional): Features keys. If None, assigned as "input_i"
                for i-th feature. Defaults to None.
            load_kwargs (dict, optional): Additional loading kwargs. Defaults to {}.

        Returns:
            DatasetType: dataset
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def assign_feature_value(
        dataset: DatasetType, feature_key: str, value: int
    ) -> DatasetType:
        """Assign a value to a feature for every sample in a Dataset

        Args:
            dataset (DatasetType): Dataset to assign the value to
            feature_key (str): Feature to assign the value to
            value (int): Value to assign

        Returns:
            DatasetType: updated dataset
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_feature_from_ds(dataset: DatasetType, feature_key: str) -> np.ndarray:
        """Get a feature from a Dataset

        Args:
            dataset (DatasetType): Dataset to get the feature from
            feature_key (str): Feature value to get

        Returns:
            np.ndarray: Feature values for dataset
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
    def has_feature_key(dataset: DatasetType, key: str) -> bool:
        """Check if a Dataset has a feature denoted by key

        Args:
            dataset (DatasetType): Dataset to check
            key (str): Key to check

        Returns:
            bool: If the dataset has a feature denoted by key
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
    def merge(
        id_dataset: DatasetType,
        ood_dataset: DatasetType,
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
    ) -> DatasetType:
        """Merge two datasets

        Args:
            id_dataset (Dataset): dataset of in-distribution data
            ood_dataset (DictDataset): dataset of out-of-distribution data
            resize (Optional[bool], optional): toggles if input tensors of the
                datasets have to be resized to have the same shape. Defaults to True.
            shape (Optional[Tuple[int]], optional): shape to use for resizing input
                tensors. If None, the tensors are resized with the shape of the
                id_dataset input tensors. Defaults to None.

        Returns:
            DatasetType: merged dataset
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def prepare_for_training(
        cls,
        dataset: DatasetType,
        batch_size: int,
        shuffle: bool = False,
        preprocess_fn: Optional[Callable] = None,
        augment_fn: Optional[Callable] = None,
        output_keys: list = ["input", "label"],
    ) -> DatasetType:
        """Prepare a dataset for training

        Args:
            dataset (DictDataset): Dataset to prepare
            batch_size (int): Batch size
            shuffle (bool): Wether to shuffle the dataloader or not
            preprocess_fn (Callable, optional): Preprocessing function to apply to
                the dataset. Defaults to None.
            augment_fn (Callable, optional): Augment function to be used (when the
                returned dataset is to be used for training). Defaults to None.
            output_keys (list): List of keys corresponding to the features that will be
                returned. Keep all features if None. Defaults to None.

        Returns:
            DatasetType: prepared dataset / dataloader
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
