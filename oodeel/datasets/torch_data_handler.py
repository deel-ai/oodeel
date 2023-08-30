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
from typing import get_args

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import default_collate

from ..types import Any
from ..types import Callable
from ..types import ItemType
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Tuple
from ..types import Union
from .data_handler import DataHandler


def dict_only_ds(ds_handling_method: Callable) -> Callable:
    """Decorator to ensure that the dataset is a dict dataset and that the input key
    matches one of the feature keys. The signature of decorated functions
    must be function(dataset, *args, **kwargs) with feature_key either in kwargs or
    args[0] when relevant.


    Args:
        ds_handling_method: method to decorate

    Returns:
        decorated method
    """

    def wrapper(dataset: Dataset, *args, **kwargs):
        assert isinstance(
            dataset, DictDataset
        ), "Dataset must be an instance of DictDataset"

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
                    key in dataset.output_keys
                ), f"The input dataset has no feature names {key}"
        return ds_handling_method(dataset, *args, **kwargs)

    return wrapper


def to_torch(array: TensorType) -> torch.Tensor:
    """Convert an array into a torch Tensor

    Args:
        array (TensorType): array to convert

    Returns:
        torch.Tensor: converted array
    """
    if isinstance(array, np.ndarray):
        return torch.Tensor(array)
    elif isinstance(array, torch.Tensor):
        return array
    else:
        raise TypeError("Input array must be of numpy or torch type")


class DictDataset(Dataset):
    r"""Dictionary pytorch dataset

    Wrapper to output a dictionary of tensors at the __getitem__ call of a dataset.
    Some mapping, filtering and concatenation methods are implemented to imitate
    tensorflow datasets features.

    Args:
        dataset (Dataset): Dataset to wrap.
        output_keys (output_keys[str]): Keys describing the output tensors.
    """

    def __init__(
        self, dataset: Dataset, output_keys: List[str] = ["input", "label"]
    ) -> None:
        self._dataset = dataset
        self._raw_output_keys = output_keys
        self.map_fns = []
        self._check_init_args()

    @property
    def output_keys(self) -> list:
        """Get the list of keys in a dict-based item from the dataset.

        Returns:
            list: feature keys of the dataset.
        """
        dummy_item = self[0]
        return list(dummy_item.keys())

    @property
    def output_shapes(self) -> list:
        """Get a list of the tensor shapes in an item from the dataset.

        Returns:
            list: tensor shapes of an dataset item.
        """
        dummy_item = self[0]
        return [dummy_item[key].shape for key in self.output_keys]

    def _check_init_args(self) -> None:
        """Check validity of dataset and output keys provided at init"""
        dummy_item = self._dataset[0]
        assert isinstance(
            dummy_item, (tuple, dict, list, torch.Tensor)
        ), "Dataset to be wrapped needs to return tuple, list or dict of tensors"
        if isinstance(dummy_item, torch.Tensor):
            dummy_item = [dummy_item]
        assert len(dummy_item) == len(
            self._raw_output_keys
        ), "Length mismatch between dataset item and provided keys"

    def __getitem__(self, index: int) -> dict:
        """Return a dictionary of tensors corresponding to a specfic index.

        Args:
            index (int): the index of the item to retrieve.

        Returns:
            dict: tensors for the item at the specific index.
        """
        item = self._dataset[index]

        # convert item to a list / tuple of tensors
        if isinstance(item, torch.Tensor):
            tensors = [item]
        elif isinstance(item, dict):
            tensors = list(item.values())
        else:
            tensors = item

        # build output dictionary
        output_dict = {
            key: tensor for (key, tensor) in zip(self._raw_output_keys, tensors)
        }

        # apply map functions
        for map_fn in self.map_fns:
            output_dict = map_fn(output_dict)
        return output_dict

    def map(self, map_fn: Callable, inplace: bool = False) -> "DictDataset":
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

    def filter(self, filter_fn: Callable, inplace: bool = False) -> "DictDataset":
        """Filter the dataset

        Args:
            filter_fn (Callable): filter function f: dict -> bool
            inplace (bool): if False, applies the filtering on a copied version of\
                the dataset. Defaults to False.

        Returns:
            DictDataset: Filtered dataset
        """
        indices = [i for i in range(len(self)) if filter_fn(self[i])]
        dataset = self if inplace else copy.deepcopy(self)
        dataset._dataset = Subset(self._dataset, indices)
        return dataset

    def concatenate(
        self, other_dataset: Dataset, inplace: bool = False
    ) -> "DictDataset":
        """Concatenate with another dataset

        Args:
            other_dataset (DictDataset): Dataset to concatenate with
            inplace (bool): if False, applies the filtering on a copied version of\
                the dataset. Defaults to False.

        Returns:
            DictDataset: Concatenated dataset
        """
        assert isinstance(
            other_dataset, DictDataset
        ), "Second dataset should be an instance of DictDataset"
        assert (
            self.output_keys == other_dataset.output_keys
        ), "Incompatible dataset elements (different dict keys)"
        if inplace:
            dataset_copy = copy.deepcopy(self)
            self._raw_output_keys = self.output_keys
            self.map_fns = []
            self._dataset = ConcatDataset([dataset_copy, other_dataset])
            dataset = self
        else:
            dataset = DictDataset(
                ConcatDataset([self, other_dataset]), self.output_keys
            )
        return dataset

    def __len__(self) -> int:
        """Return the length of the dataset, i.e. the number of items.

        Returns:
            int: length of the dataset.
        """
        return len(self._dataset)


class TorchDataHandler(DataHandler):
    """
    Class to manage torch DictDataset. The aim is to provide a simple interface
    for working with torch datasets and manage them without having to use
    torch syntax.
    """

    @staticmethod
    def _default_target_transform(y: Any) -> torch.Tensor:
        """Format int or float item target as a torch tensor

        Args:
            y (Any): dataset item target

        Returns:
            torch.Tensor: target as a torch.Tensor
        """
        return torch.tensor(y) if isinstance(y, (float, int)) else y

    DEFAULT_TRANSFORM = torchvision.transforms.PILToTensor()
    DEFAULT_TARGET_TRANSFORM = _default_target_transform.__func__

    @classmethod
    def load_dataset(
        cls,
        dataset_id: Union[Dataset, ItemType, str],
        keys: Optional[list] = None,
        load_kwargs: dict = {},
    ) -> DictDataset:
        """Load dataset from different manners

        Args:
            dataset_id (Union[Dataset, ItemType, str]): dataset identification
            keys (list, optional): Features keys. If None, assigned as "input_i"
                for i-th feature. Defaults to None.
            load_kwargs (dict, optional): Additional loading kwargs. Defaults to {}.

        Returns:
            DictDataset: dataset
        """
        if isinstance(dataset_id, str):
            assert "root" in load_kwargs.keys()
            dataset = cls.load_from_torchvision(dataset_id, **load_kwargs)
        elif isinstance(dataset_id, Dataset):
            dataset = cls.load_custom_dataset(dataset_id, keys)
        elif isinstance(dataset_id, get_args(ItemType)):
            dataset = cls.load_dataset_from_arrays(dataset_id, keys)
        return dataset

    @staticmethod
    def load_dataset_from_arrays(
        dataset_id: ItemType,
        keys: Optional[list] = None,
    ) -> DictDataset:
        """Load a torch.utils.data.Dataset from an array or a tuple/dict of arrays.

        Args:
            dataset_id (ItemType):
                numpy / torch array(s) to load.
            keys (list, optional): Features keys. If None, assigned as "input_i"
                for i-th feature. Defaults to None.

        Returns:
            DictDataset: dataset
        """
        # If dataset_id is an array
        if isinstance(dataset_id, get_args(TensorType)):
            tensors = tuple(to_torch(dataset_id))
            output_keys = keys or ["input"]

        # If dataset_id is a tuple of arrays
        elif isinstance(dataset_id, tuple):
            len_elem = len(dataset_id)
            output_keys = keys
            if output_keys is None:
                if len_elem == 2:
                    output_keys = ["input", "label"]
                else:
                    output_keys = [f"input_{i}" for i in range(len_elem - 1)] + [
                        "label"
                    ]
                    print(
                        "Loading torch.utils.data.Dataset with elems as dicts, "
                        'assigning "input_i" key to the i-th tuple dimension and'
                        ' "label" key to the last tuple dimension.'
                    )
            assert len(output_keys) == len(dataset_id)
            tensors = tuple(to_torch(array) for array in dataset_id)

        # If dataset_id is a dictionary of arrays
        elif isinstance(dataset_id, dict):
            output_keys = keys or list(dataset_id.keys())
            assert len(output_keys) == len(dataset_id)
            tensors = tuple(to_torch(array) for array in dataset_id.values())

        # create torch dictionary dataset from tensors tuple and keys
        dataset = DictDataset(TensorDataset(*tensors), output_keys)
        return dataset

    @staticmethod
    def load_custom_dataset(
        dataset_id: Dataset, keys: Optional[list] = None
    ) -> DictDataset:
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
            output_keys = keys
            if output_keys is None:
                len_elem = len(dummy_item)
                if len_elem == 2:
                    output_keys = ["input", "label"]
                else:
                    output_keys = [f"input_{i}" for i in range(len_elem - 1)] + [
                        "label"
                    ]
                    print(
                        "Feature name not found, assigning 'input_i' "
                        "key to the i-th tensor and 'label' key to the last"
                    )
            dataset_id = DictDataset(dataset_id, output_keys)

        dataset = dataset_id
        return dataset

    @classmethod
    def load_from_torchvision(
        cls,
        dataset_id: str,
        root: str,
        transform: Callable = DEFAULT_TRANSFORM,
        target_transform: Callable = DEFAULT_TARGET_TRANSFORM,
        download: bool = False,
        **load_kwargs,
    ) -> DictDataset:
        """Load a Dataset from the torchvision datasets catalog

        Args:
            dataset_id (str): Identifier of the dataset
            root (str): Root directory of dataset
            transform (Callable, optional): Transform function to apply to the input.
                Defaults to DEFAULT_TRANSFORM.
            target_transform (Callable, optional): Transform function to apply
                to the target. Defaults to DEFAULT_TARGET_TRANSFORM.
            download (bool):  If true, downloads the dataset from the internet and puts
                it in root directory. If dataset is already downloaded, it is not
                downloaded again. Defaults to False.
            load_kwargs (dict): Loading kwargs to add to the initialization
                of dataset.

        Returns:
            DictDataset: dataset
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
        return cls.load_custom_dataset(dataset)

    @staticmethod
    def assign_feature_value(
        dataset: DictDataset, feature_key: str, value: int
    ) -> DictDataset:
        """Assign a value to a feature for every sample in a DictDataset

        Args:
            dataset (DictDataset): DictDataset to assign the value to
            feature_key (str): Feature to assign the value to
            value (int): Value to assign

        Returns:
            DictDataset
        """
        assert isinstance(
            dataset, DictDataset
        ), "Dataset must be an instance of DictDataset"

        def assign_value_to_feature(x):
            x[feature_key] = torch.tensor(value)
            return x

        dataset = dataset.map(assign_value_to_feature)
        return dataset

    @staticmethod
    @dict_only_ds
    def get_feature_from_ds(dataset: DictDataset, feature_key: str) -> np.ndarray:
        """Get a feature from a DictDataset

        !!! note
            This function can be a bit time consuming since it needs to iterate
            over the whole dataset.

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
    @dict_only_ds
    def get_ds_feature_keys(dataset: DictDataset) -> list:
        """Get the feature keys of a DictDataset

        Args:
            dataset (DictDataset): Dataset to get the feature keys from

        Returns:
            list: List of feature keys
        """
        return dataset.output_keys

    @staticmethod
    def has_feature_key(dataset: DictDataset, key: str) -> bool:
        """Check if a DictDataset has a feature denoted by key

        Args:
            dataset (DictDataset): Dataset to check
            key (str): Key to check

        Returns:
            bool: If the dataset has a feature denoted by key
        """
        assert isinstance(
            dataset, DictDataset
        ), "Dataset must be an instance of DictDataset"

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
    @dict_only_ds
    def filter_by_feature_value(
        dataset: DictDataset,
        feature_key: str,
        values: list,
        excluded: bool = False,
    ) -> DictDataset:
        """Filter the dataset by checking the value of a feature is in `values`

        !!! note
            This function can be a bit of time consuming since it needs to iterate
            over the whole dataset.

        Args:
            dataset (DictDataset): Dataset to filter
            feature_key (str): Feature name to check the value
            values (list): Feature_key values to keep
            excluded (bool, optional): To keep (False) or exclude (True) the samples
                with Feature_key value included in Values. Defaults to False.

        Returns:
            DictDataset: Filtered dataset
        """

        if len(dataset[0][feature_key].shape) > 0:
            value_dim = dataset[0][feature_key].shape[-1]
            values = [
                F.one_hot(torch.tensor(value).long(), value_dim) for value in values
            ]

        def filter_fn(x):
            keep = any([torch.all(x[feature_key] == v) for v in values])
            return keep if not excluded else not keep

        filtered_dataset = dataset.filter(filter_fn)
        return filtered_dataset

    @classmethod
    def prepare_for_training(
        cls,
        dataset: DictDataset,
        batch_size: int,
        shuffle: bool = False,
        preprocess_fn: Optional[Callable] = None,
        augment_fn: Optional[Callable] = None,
        output_keys: Optional[list] = None,
        dict_based_fns: bool = False,
        shuffle_buffer_size: Optional[int] = None,
    ) -> DataLoader:
        """Prepare a DataLoader for training

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
            dict_based_fns (bool): Whether to use preprocess and DA functions as dict
                based (if True) or as tuple based (if False). Defaults to False.
            shuffle_buffer_size (int, optional): Size of the shuffle buffer. Not used
                in torch because we only rely on Map-Style datasets. Still as argument
                for API consistency. Defaults to None.

        Returns:
            DataLoader: dataloader
        """
        output_keys = output_keys or cls.get_ds_feature_keys(dataset)

        def collate_fn(batch: List[dict]):
            if dict_based_fns:
                # preprocess + DA: List[dict] -> List[dict]
                preprocess_func = preprocess_fn or (lambda x: x)
                augment_func = augment_fn or (lambda x: x)
                batch = [augment_func(preprocess_func(d)) for d in batch]
                # to tuple of batchs
                return tuple(
                    default_collate([d[key] for d in batch]) for key in output_keys
                )
            else:
                # preprocess + DA: List[dict] -> List[tuple]
                preprocess_func = preprocess_fn or (lambda *x: x)
                augment_func = augment_fn or (lambda *x: x)
                batch = [
                    augment_func(
                        *preprocess_func(*tuple(d[key] for key in output_keys))
                    )
                    for d in batch
                ]
                # to tuple of batchs
                return default_collate(batch)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
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
        # If a desired shape is given, triggers the resize
        if shape is not None:
            resize = True

        # If the shape of the two datasets are different, triggers the resize
        if id_dataset.output_shapes[0] != ood_dataset.output_shapes[0]:
            resize = True
            if shape is None:
                print(
                    "Resizing the first item of elem (usually the image)",
                    " with the shape of id_dataset",
                )
                shape = id_dataset.output_shapes[0][1:]

        if resize:
            resize_fn = torchvision.transforms.Resize(shape)

            def reshape_fn(item_dict):
                item_dict["input"] = resize_fn(item_dict["input"])
                return item_dict

            id_dataset = id_dataset.map(reshape_fn)
            ood_dataset = ood_dataset.map(reshape_fn)

        merged_dataset = id_dataset.concatenate(ood_dataset)
        return merged_dataset

    @staticmethod
    def get_item_length(dataset: Dataset) -> int:
        """Number of elements in a dataset item

        Args:
            dataset (DictDataset): Dataset

        Returns:
            int: Item length
        """
        return len(dataset[0])

    @staticmethod
    def get_dataset_length(dataset: Dataset) -> int:
        """Number of items in a dataset

        Args:
            dataset (DictDataset): Dataset

        Returns:
            int: Dataset length
        """
        return len(dataset)

    @staticmethod
    def get_feature_shape(dataset: Dataset, feature_key: Union[str, int]) -> tuple:
        """Get the shape of a feature of dataset identified by feature_key

        Args:
            dataset (Dataset): a Dataset
            feature_key (Union[str, int]): The identifier of the feature

        Returns:
            tuple: the shape of feature_id
        """
        return tuple(dataset[0][feature_key].shape)

    @staticmethod
    def get_input_from_dataset_item(elem: ItemType) -> Any:
        """Get the tensor that is to be feed as input to a model from a dataset element.

        Args:
            elem (ItemType): dataset element to extract input from

        Returns:
            Any: Input tensor
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
        label = item[1]  # labels must be at index 1 in the batch tuple
        # If labels are one-hot encoded, take the argmax
        if len(label.shape) > 1 and label.shape[1] > 1:
            label = label.view(label.size(0), -1)
            label = torch.argmax(label, dim=1)
        # If labels are in two dimensions, squeeze them
        if len(label.shape) > 1:
            label = label.view([label.shape[0]])
        return label

    @staticmethod
    def get_feature(dataset: DictDataset, feature_key: Union[str, int]) -> DictDataset:
        """Extract a feature from a dataset

        Args:
            dataset (tf.data.Dataset): Dataset to extract the feature from
            feature_key (Union[str, int]): feature to extract

        Returns:
            tf.data.Dataset: dataset built with the extracted feature only
        """

        def _get_feature_item(item):
            return item[feature_key]

        return dataset.map(_get_feature_item)
