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
import copy

import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from tqdm import tqdm

from ..types import Any
from ..types import ArrayLike
from ..types import Callable
from ..types import Dict
from ..types import List
from ..types import Optional
from ..types import Tuple
from ..types import Union
from .data_handler import DataHandler


DEFAULT_TRANSFORM = torchvision.transforms.ToTensor()
DEFAULT_TARGET_TRANSFORM = lambda y: torch.LongTensor(
    [y] if isinstance(y, (float, int)) else y
)


def to_torch(array: ArrayLike):
    """Convert an array into a torch Tensor"""
    if isinstance(array, np.ndarray):
        return torch.Tensor(array)
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise TypeError("Input array must be of numpy or torch type")


class DictDataset(Dataset):
    r"""Dictionary pytorch dataset

    Wrapper to output a dictionary of tensors instead of a tuple at the __getitem__ call
    of a dataset. Some mapping, filtering and concatenation methods are implemented to
    imitate tensorflow datasets features.

    Args:
        dataset (Dataset): Dataset to wrap.
        output_keys (output_keys[str]): Keys describing the output tensors.
    """

    def __init__(
        self, dataset: Dataset, output_keys: List[str] = ["input", "label"]
    ) -> None:
        self._dataset = dataset
        self._raw_output_keys = output_keys
        self._added_output_keys = []
        self.map_fns = []
        self._check_keys_and_dataset()

    def _check_keys_and_dataset(self):
        """Check if the initialization arguments are correct"""
        assert isinstance(
            self._dataset[0], (Tuple, List, torch.Tensor)
        ), "Dataset to be wrapped needs to output tuple of tensors"
        assert len(self._dataset[0]) == len(
            self._raw_output_keys
        ), "Length mismatch between tuple of tensors and dictionary keys"

    @property
    def output_keys(self):
        return self._raw_output_keys + self._added_output_keys

    @property
    def output_shapes(self):
        dummy_item = self[0]
        return [dummy_item[key].shape for key in self.output_keys]

    def __getitem__(self, index):
        """Return a dictionary of tensors corresponding to a specfic index"""
        tensors = self._dataset[index]
        output_dict = {
            key: tensor for (key, tensor) in zip(self._raw_output_keys, tensors)
        }
        for map_fn in self.map_fns:
            output_dict = map_fn(output_dict)
        return output_dict

    def map(self, map_fn: Callable, inplace: bool = False):
        """Map the dataset

        Args:
            map_fn (Callable): map function f: dict -> dict
            inplace (bool): if False, applies the mapping on a copied version of\
                the dataset. Defaults to False.

        Return:
            DictDataset: Mapped dataset
        """
        dataset = self if inplace else copy.deepcopy(self)
        dataset.map_fns.append(map_fn)
        return dataset

    def filter(self, filter_fn: Callable, inplace: bool = False):
        """Filter the dataset

        Args:
            filter_fn (Callable): filter function f: dict -> bool
            inplace (bool): if False, applies the filtering on a copied version of\
                the dataset. Defaults to False.

        Returns:
            DictDataset: Filtered dataset
        """
        indices = [
            i
            for i in tqdm(range(len(self)), desc="Filtering the dataset...")
            if filter_fn(self[i])
        ]
        dataset = self if inplace else copy.deepcopy(self)
        dataset._dataset = Subset(self._dataset, indices)
        return dataset

    def concatenate(self, other_dataset: Dataset, inplace: bool = False):
        """Concatenate with another dataset

        !!! warning
            The map functions that will be retained will be those of this dataset

        Args:
            other_dataset (Dataset): Dataset to concatenate with
            inplace (bool): if False, applies the filtering on a copied version of\
                the dataset. Defaults to False.

        Returns:
            DictDataset: Concatenated dataset
        """
        dataset = self if inplace else copy.deepcopy(self)
        dataset._dataset = ConcatDataset([self._dataset, other_dataset])
        return dataset

    def __len__(self):
        return len(self._dataset)


class TorchDataHandler(DataHandler):
    """
    Class to manage tf.data.Dataset. The aim is to provide a simple interface for
    working with tf.data.Datasets and manage them without having to use
    tensorflow syntax.
    """

    @staticmethod
    def load_dataset(dataset_id: Any, load_kwargs: dict = {}):
        """Load dataset from different manners

        Args:
            dataset_id (Any): dataset identification
            load_kwargs (dict, optional): Additional loading kwargs. Defaults to {}.

        Returns:
            Any: dataset
        """
        if isinstance(dataset_id, str):
            dataset = TorchDataHandler.load_torchvision_dataset(
                dataset_id, **load_kwargs
            )
        elif isinstance(dataset_id, Dataset):
            dataset = TorchDataHandler.load_custom_dataset(dataset_id)
        else:
            dataset = TorchDataHandler.load_dataset_from_arrays(dataset_id)
        return dataset

    @staticmethod
    def load_dataset_from_arrays(
        dataset_id: Union[
            ArrayLike,
            Dict[str, ArrayLike],
            Tuple[ArrayLike],
        ],
    ) -> DictDataset:
        """Load a torch.utils.data.Dataset from an array or a tuple/dict of arrays.

        Args:
            dataset_id (ArrayLike | Dict[str, ArrayLike] | Tuple[ArrayLike]):\
                numpy / torch array(s) to load.

        Returns:
            DictDataset
        """
        # If dataset_id is an array
        if isinstance(dataset_id, (np.ndarray, torch.Tensor)):
            tensors = tuple(to_torch(dataset_id))
            output_keys = ["input"]

        # If dataset_id is a tuple of arrays
        elif isinstance(dataset_id, tuple):
            len_elem = len(dataset_id)
            if len_elem == 2:
                output_keys = ["input", "label"]
            else:
                output_keys = [f"input_{i}" for i in range(len_elem - 1)] + ["label"]
                print(
                    "Loading torch.utils.data.Dataset with elems as dicts, assigning "
                    '"input_i" key to the i-th tuple dimension and "label" key to '
                    "the last tuple dimension."
                )
            tensors = tuple(to_torch(array) for array in dataset_id)

        # If dataset_id is a dictionary of arrays
        elif isinstance(dataset_id, dict):
            output_keys = list(dataset_id.keys())
            tensors = tuple(to_torch(dataset_id[k]) for k in output_keys)

        # create torch dictionary dataset from tensors tuple and keys
        dataset = DictDataset(TensorDataset(*tensors), output_keys)
        return dataset

    # TODO: adapt DictDataset to read dict based custom datasets
    @staticmethod
    def load_custom_dataset(dataset_id: Dataset, keys: list = None) -> DictDataset:
        """Load a custom Dataset by ensuring it has the correct format (dict-based)

        Args:
            dataset_id (Dataset): Dataset
            keys (list, optional): Keys to use for features if dataset_id is
                tuple based. Defaults to None.

        Returns:
            DictDataset
        """
        # If dataset_id is a tuple based Dataset, convert it to a DictDataset
        dummy_item = dataset_id[0]
        if not isinstance(dummy_item, dict):
            assert isinstance(
                dummy_item, (Tuple, torch.Tensor)
            ), "Custom dataset should be either dictionary based or tuple based"
            if keys is None:
                print(
                    "Feature name not found, assigning 'input_i' "
                    "key to the i-th tensor and 'label' key to the last"
                )
                len_elem = len(dummy_item)
                if len_elem == 2:
                    output_keys = ["input", "label"]
                else:
                    output_keys = [f"input_{i}" for i in range(len_elem - 1)] + [
                        "label"
                    ]
            dataset_id = DictDataset(dataset_id, output_keys)

        dataset = dataset_id
        return dataset

    @staticmethod
    def load_torchvision_dataset(
        dataset_id: str,
        root: str,
        transform: Callable = DEFAULT_TRANSFORM,
        target_transform: Callable = DEFAULT_TARGET_TRANSFORM,
        download: bool = False,
        load_kwargs: dict = {},
    ) -> DictDataset:
        """Load a Dataset from the torchvision datasets catalog

        Args:
            dataset_id (str): Identifier of the dataset
            root (str): Root directory of dataset
            download (bool):  If true, downloads the dataset from the internet and puts\
                it in root directory. If dataset is already downloaded, it is not\
                downloaded again. Defaults to False.
            load_kwargs (dict, optional): Loading kwargs to add to the initialization\
                of dataset. Defaults to {}.

        Returns:
            DictDataset
        """
        assert (
            dataset_id in torchvision.datasets.__all__
        ), "Dataset not available on torchvision datasets catalog"
        dataset = getattr(torchvision.datasets, dataset_id)(
            root=root,
            download=download,
            transform=transform,
            target_transform=target_transform,
            **load_kwargs,
        )
        return TorchDataHandler.load_custom_dataset(dataset)

    @staticmethod
    def assign_feature_value(
        dataset: DictDataset, feature_key: str, value: int
    ) -> DictDataset:
        """Assign a value to a feature for every samples in a DictDataset

        Args:
            dataset (DictDataset): DictDataset to assigne the value to
            feature_key (str): Feature to assign the value to
            value (int): Value to assign

        Returns:
            DictDataset
        """
        assert isinstance(
            dataset, DictDataset
        ), "Dataset must be an instance of DictDataset"

        def assign_value_to_feature(x):
            x[feature_key] = value
            return x

        dataset = dataset.map(assign_value_to_feature)
        if feature_key not in dataset.output_keys:
            dataset._added_output_keys += [feature_key]
        return dataset

    @staticmethod
    def get_feature_from_ds(dataset: DictDataset, feature_key: str) -> np.ndarray:
        """Get a feature from a DictDataset

        Args:
            dataset (DictDataset): Dataset to get the feature from
            feature_key (str): Feature value to get

        Returns:
            np.ndarray: Feature values for dataset
        """
        features = dataset.map(lambda x: x[feature_key])
        features = np.stack([f.numpy() for f in features])
        return features

    @staticmethod
    def get_ds_feature_keys(dataset: DictDataset) -> list:
        """Get the feature keys of a DictDataset

        Args:
            dataset (DictDataset): Dataset to get the feature keys from

        Returns:
            list: List of feature keys
        """
        return dataset.output_keys

    @staticmethod
    def has_key(dataset: DictDataset, key: str) -> bool:
        """Check if a DictDataset has a feature denoted by key

        Args:
            dataset (DictDataset): Dataset to check
            key (str): Key to check

        Returns:
            bool: If the dataset has a feature denoted by key
        """
        assert isinstance(
            dataset, DictDataset
        ), "dataset must be an instance of DictDataset"
        return key in dataset.output_keys

    @staticmethod
    def map_ds(
        dataset: DictDataset,
        map_fn: Callable,
    ) -> DictDataset:
        """Map a function to a DictDataset

        Args:
            dataset (DictDataset): Dataset to map the function to
            map_fn (Callable): Function to map

        Returns:
            DictDataset: Mapped dataset
        """
        return dataset.map(map_fn)

    @staticmethod
    def filter_by_feature_value(
        dataset: DictDataset,
        feature_key: str,
        values: list,
    ):
        """Filter the dataset by checking the value of a feature is in `values`

        Args:
            dataset (DictDataset): Dataset to filter
            feature_key (str): Feature name to check the value
            values (list): Feature_key values to keep (if excluded is False)
                or to exclude

        Returns:
            DictDataset: Filtered dataset
        """
        filtered_dataset = dataset.filter(lambda x: x[feature_key] in values)
        return filtered_dataset

    @staticmethod
    def prepare_for_training(
        dataset: DictDataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        """Prepare a DataLoader for training

        Args:
            dataset (DictDataset): Dataset to prepare
            batch_size (int): Batch size
            shuffle (bool): Wether to shuffle the dataloader or not

        Returns:
            DataLoader: dataloader
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    @staticmethod
    def merge(
        id_dataset: DictDataset,
        ood_dataset: DictDataset,
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
    ) -> DictDataset:
        """Merge two instances of DictDataset

        Args:
            id_dataset (DictDataset): dataset of in-distribution data
            ood_dataset (DictDataset): dataset of out-of-distribution data
            resize (Optional[bool], optional): toggles if input tensors of the
                datasets have to be resized to have the same shape. Defaults to True.
            shape (Optional[Tuple[int]], optional): shape to use for resizing input
                tensors. If None, the tensors are resized with the shape of the
                id_dataset input tensors. Defaults to None.

        Returns:
            DictDataset: merged dataset
        """
        assert (
            id_dataset.output_keys == ood_dataset.output_keys
        ), "incompatible dataset elements (different dict keys)"

        # If a desired shape is given, triggers the resize
        if shape is not None:
            resize = True

        # If the shape of the two datasets are different, triggers the resize
        if id_dataset.output_shapes != ood_dataset.output_shapes:
            resize = True
            if shape is None:
                print(
                    "Resizing the first item of elem (usually the image)",
                    " with the shape of id_dataset",
                )
                shape = id_dataset.output_shapes[:-1]

        if resize:
            resize_fn = torchvision.transforms.Resize(shape)

            def reshape_fn(item_dict):
                item_dict["input"] = resize_fn(item_dict["input"])
                return item_dict

            id_dataset = id_dataset.map(reshape_fn)
            ood_dataset = ood_dataset.map(reshape_fn)

        merged_dataset = id_dataset.concatenate(ood_dataset)
        return merged_dataset
