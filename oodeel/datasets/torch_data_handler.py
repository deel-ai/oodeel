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
from datasets import load_dataset as hf_load_dataset
from matplotlib import transforms
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

    def wrapper(dataset: Dataset, *args, **kwargs):
        assert isinstance(
            dataset, DictDataset
        ), "Dataset must be an instance of DictDataset"

        if "column_name" in kwargs:
            column_name = kwargs["column_name"]
        elif len(args) > 0:
            column_name = args[0]

        # If column_name is provided, check that it is in the dataset column names
        if (len(args) > 0) or ("column_name" in kwargs):
            if isinstance(column_name, str):
                column_name = [column_name]
            for name in column_name:
                assert (
                    name in dataset.column_names
                ), f"The input dataset has no column named {name}"
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
        columns (columns[str]): Column names describing the output tensors.
    """

    def __init__(
        self, dataset: Dataset, column_names: List[str] = ["input", "label"]
    ) -> None:
        self._dataset = dataset
        self._raw_columns = column_names
        self.map_fns = []
        self._check_init_args()

    @property
    def column_names(self) -> list:
        """Get the list of columns in a dict-based item from the dataset.

        Returns:
            list: column names of the dataset.
        """
        dummy_item = self[0]
        return list(dummy_item.keys())

    def _check_init_args(self) -> None:
        """Check validity of dataset and column names provided at init"""
        dummy_item = self._dataset[0]
        assert isinstance(
            dummy_item, (tuple, dict, list, torch.Tensor)
        ), "Dataset to be wrapped needs to return tuple, list or dict of tensors"
        if isinstance(dummy_item, torch.Tensor):
            dummy_item = [dummy_item]
        assert len(dummy_item) == len(
            self._raw_columns
        ), "Length mismatch between dataset item and provided column names"

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
        output_dict = {key: tensor for (key, tensor) in zip(self._raw_columns, tensors)}

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

    def __init__(self) -> None:
        """
        Initializes the TorchDataHandler.
        Attributes:
            backend (str): The backend framework used, set to "torch".
            channel_order (str): The channel order format, set to "channels_first".
        """

        super().__init__()
        self.backend = "torch"
        self.channel_order = "channels_first"

    @staticmethod
    def _default_target_transform(y: Any) -> torch.Tensor:
        """Format int or float item target as a torch tensor

        Args:
            y (Any): dataset item target

        Returns:
            torch.Tensor: target as a torch.Tensor
        """
        return torch.tensor(y) if isinstance(y, (float, int)) else y

    def load_dataset(
        cls,
        dataset_id: Union[Dataset, ItemType, str],
        columns: Optional[list] = None,
        hub: Optional[str] = "torchvision",
        load_kwargs: dict = {},
    ) -> DictDataset:
        """Load dataset from different manners

        Args:
            dataset_id (Union[Dataset, ItemType, str]): dataset identification.
            Can be the name of a dataset from torchvision, a torch Dataset,
            or a tuple/dict of np.ndarrays/torch tensors.
            columns (list, optional): Column names. If None, assigned as "input_i"
                for i-th feature. Defaults to None.
            load_kwargs (dict, optional): Additional loading kwargs. Defaults to {}.

        Returns:
            DictDataset: dataset
        """

        assert hub in {
            "torchvision",
            "huggingface",
        }, "hub must be either 'torchvision' or 'huggingface'"

        if isinstance(dataset_id, str):
            if hub == "torchvision":
                assert "root" in load_kwargs.keys()
                dataset = cls.load_from_torchvision(dataset_id, load_kwargs)
            elif hub == "huggingface":
                dataset = cls.load_from_huggingface(dataset_id, load_kwargs)
        elif isinstance(dataset_id, Dataset):
            dataset = cls.load_custom_dataset(dataset_id, columns)
        elif isinstance(dataset_id, get_args(ItemType)):
            dataset = cls.load_dataset_from_arrays(dataset_id, columns)
        return dataset

    @staticmethod
    def load_dataset_from_arrays(
        dataset_id: ItemType,
        columns: Optional[list] = None,
    ) -> DictDataset:
        """Load a torch.utils.data.Dataset from an array or a tuple/dict of arrays.

        Args:
            dataset_id (ItemType):
                numpy / torch array(s) to load.
            columns (list, optional): Column names to assign. If None,
                assigned as "input_i" for i-th feature. Defaults to None.

        Returns:
            DictDataset: dataset
        """
        # If dataset_id is an array
        if isinstance(dataset_id, get_args(TensorType)):
            tensors = tuple(to_torch(dataset_id))
            columns = columns or ["input"]

        # If dataset_id is a tuple of arrays
        elif isinstance(dataset_id, tuple):
            len_elem = len(dataset_id)
            if columns is None:
                if len_elem == 2:
                    columns = ["input", "label"]
                else:
                    columns = [f"input_{i}" for i in range(len_elem - 1)] + ["label"]
                    print(
                        "Loading torch.utils.data.Dataset with elems as dicts, "
                        'assigning "input_i" key to the i-th tuple dimension and'
                        ' "label" key to the last tuple dimension.'
                    )
            assert len(columns) == len(dataset_id)
            tensors = tuple(to_torch(array) for array in dataset_id)

        # If dataset_id is a dictionary of arrays
        elif isinstance(dataset_id, dict):
            columns = columns or list(dataset_id.keys())
            assert len(columns) == len(dataset_id)
            tensors = tuple(to_torch(array) for array in dataset_id.values())

        # create torch dictionary dataset from tensors tuple and columns
        dataset = DictDataset(TensorDataset(*tensors), columns)
        return dataset

    @staticmethod
    def load_custom_dataset(
        dataset_id: Dataset, columns: Optional[list] = None
    ) -> DictDataset:
        """Load a custom Dataset by ensuring it has the correct format (dict-based)

        Args:
            dataset_id (Dataset): Dataset
            columns (list, optional): Column names to use for elements if dataset_id is
                tuple based. If None, assigned as "input_i"
                for i-th column. Defaults to None.

        Returns:
            DictDataset
        """
        # If dataset_id is a tuple based Dataset, convert it to a DictDataset
        dummy_item = dataset_id[0]
        if not isinstance(dummy_item, dict):
            assert isinstance(
                dummy_item, (Tuple, torch.Tensor)
            ), "Custom dataset should be either dictionary based or tuple based"
            if columns is None:
                len_elem = len(dummy_item)
                if len_elem == 2:
                    columns = ["input", "label"]
                else:
                    columns = [f"input_{i}" for i in range(len_elem - 1)] + ["label"]
                    print(
                        "Feature name not found, assigning 'input_i' "
                        "key to the i-th tensor and 'label' key to the last"
                    )
            dataset_id = DictDataset(dataset_id, columns)

        dataset = dataset_id
        return dataset

    @classmethod
    def load_from_huggingface(
        cls,
        dataset_id: str,
        load_kwargs: dict = {},
    ) -> DictDataset:
        """Load a Dataset from the Hugging Face datasets catalog

        Args:
            dataset_id (str): Identifier of the dataset
            load_kwargs (dict): Loading kwargs to add to the initialization
            of the dataset.

        Returns:
            DictDataset: dataset
        """
        dataset = hf_load_dataset(dataset_id, **load_kwargs)

        if "transform" not in load_kwargs.keys() and "image" in dataset.column_names:

            def transform(examples):
                examples["image"] = [
                    torchvision.transforms.PILToTensor()(img)
                    for img in examples["image"]
                ]
                examples["label"] = [
                    cls._default_target_transform(example)
                    for example in examples["label"]
                ]
                return examples

        elif "transform" not in load_kwargs.keys():

            def transform(examples):
                examples["label"] = [
                    cls._default_target_transform(example)
                    for example in examples["label"]
                ]
                return examples

        dataset = dataset.with_transform(transform)
        return DictDataset(dataset, column_names=dataset.column_names)

    @classmethod
    def load_from_torchvision(
        cls,
        dataset_id: str,
        load_kwargs: dict = {},
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

        if "transform" not in load_kwargs.keys():
            transform = torchvision.transforms.PILToTensor()
        if "target_transform" not in load_kwargs.keys():
            target_transform = cls._default_target_transform

        dataset = getattr(torchvision.datasets, dataset_id)(
            transform=transform,
            target_transform=target_transform,
            **load_kwargs,
        )
        return cls.load_custom_dataset(dataset)

    @staticmethod
    @dict_only_ds
    def get_ds_column_names(dataset: DictDataset) -> list:
        """Get the column names of a DictDataset

        Args:
            dataset (DictDataset): Dataset to get the column names from

        Returns:
            list: List of column names
        """
        return dataset.column_names

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
    def filter_by_value(
        dataset: DictDataset,
        column_name: str,
        values: list,
        excluded: bool = False,
    ) -> DictDataset:
        """Filter the dataset by checking if the value of a column is in `values`

        !!! note
            This function can be a bit of time consuming since it needs to iterate
            over the whole dataset.

        Args:
            dataset (DictDataset): Dataset to filter
            column_name (str): Column to filter the dataset with
            values (list): Column values to keep
            excluded (bool, optional): To keep (False) or exclude (True) the samples
                with column values included in Values. Defaults to False.

        Returns:
            DictDataset: Filtered dataset
        """

        if len(dataset[0][column_name].shape) > 0:
            value_dim = dataset[0][column_name].shape[-1]
            values = [
                F.one_hot(torch.tensor(value).long(), value_dim) for value in values
            ]

        def filter_fn(x):
            keep = any([torch.all(x[column_name] == v) for v in values])
            return keep if not excluded else not keep

        filtered_dataset = dataset.filter(filter_fn)
        return filtered_dataset

    @classmethod
    def prepare(
        cls,
        dataset: DictDataset,
        batch_size: int,
        preprocess_fn: Optional[Callable] = None,
        augment_fn: Optional[Callable] = None,
        columns: Optional[list] = None,
        shuffle: bool = False,
        dict_based_fns: bool = True,
        return_tuple: bool = True,
        num_workers: int = 8,
    ) -> DataLoader:
        """Prepare a DataLoader for training

        Args:
            dataset (DictDataset): Dataset to prepare
            batch_size (int): Batch size
            preprocess_fn (Callable, optional): Preprocessing function to apply to
                the dataset. Defaults to None.
            augment_fn (Callable, optional): Augment function to be used (when the
                returned dataset is to be used for training). Defaults to None.
            columns (list, optional): List of column names corresponding to the columns
                that will be returned. Keep all features if None. Defaults to None.
            shuffle (bool, optional): To shuffle the returned dataset or not.
                Defaults to False.
            dict_based_fns (bool): Whether to use preprocess and DA functions as dict
                based (if True) or as tuple based (if False). Defaults to True.
            return_tuple (bool, optional): Whether to return each dataset item
                as a tuple. Defaults to True.
            num_workers (int, optional): Number of workers to use for the dataloader.

        Returns:
            DataLoader: dataloader
        """
        columns = columns or cls.get_ds_column_names(dataset)

        def collate_fn(batch: List[dict]):
            if dict_based_fns:
                # preprocess + DA: List[dict] -> List[dict]
                preprocess_func = preprocess_fn or (lambda x: x)
                augment_func = augment_fn or (lambda x: x)
                batch = [augment_func(preprocess_func(d)) for d in batch]
                # to dict of batchs
                if return_tuple:
                    return tuple(
                        default_collate([d[key] for d in batch]) for key in columns
                    )
                return {
                    key: default_collate([d[key] for d in batch]) for key in columns
                }
            else:
                # preprocess + DA: List[dict] -> List[tuple]
                preprocess_func = preprocess_fn or (lambda *x: x)
                augment_func = augment_fn or (lambda *x: x)
                batch = [
                    augment_func(*preprocess_func(*tuple(d[key] for key in columns)))
                    for d in batch
                ]
                # to tuple of batchs
                return default_collate(batch)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return loader

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
    def get_column_elements_shape(
        dataset: Dataset, column_name: Union[str, int]
    ) -> tuple:
        """Get the shape of the elements of a column of dataset identified by
        column_name

        Args:
            dataset (Dataset): a Dataset
            column_name (Union[str, int]): The column name to get
                the element shape from.

        Returns:
            tuple: the shape of an element from column_name
        """
        return tuple(dataset[0][column_name].shape)

    @staticmethod
    def get_columns_shapes(dataset: Dataset) -> dict:
        """Get the shapes of the elements of all columns of a dataset

        Args:
            dataset (Dataset): a Dataset

        Returns:
            dict: dictionary of column names and their corresponding shape
        """
        shapes = {}
        for key in dataset.column_names:
            try:
                shapes[key] = tuple(dataset[0][key].shape)
            except AttributeError:
                pass
        return shapes

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
