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
import os
import tempfile

import numpy as np
import pytest
import torch
import torchvision

from oodeel.datasets import load_data_handler
from oodeel.datasets.torch_data_handler import Dataset
from oodeel.datasets.torch_data_handler import dict_only_ds
from oodeel.datasets.torch_data_handler import DictDataset
from oodeel.datasets.torch_data_handler import TorchDataHandler
from tests.tests_torch import generate_data
from tests.tests_torch import generate_data_torch


def assign_value_to_column(
    dataset: DictDataset, column_name: str, value: int
) -> DictDataset:
    """Assign a value to a column for every sample in a DictDataset

    Args:
        dataset (DictDataset): DictDataset to assign the value to
        column_name (str): Column to assign the value to
        value (int): Value to assign

    Returns:
        DictDataset
    """
    assert isinstance(
        dataset, DictDataset
    ), "Dataset must be an instance of DictDataset"

    def assign_value(x):
        x[column_name] = torch.tensor(value)
        return x

    dataset = dataset.map(assign_value)
    return dataset


def get_dataset_length(dataset: Dataset) -> int:
    """Number of items in a dataset

    Args:
        dataset (DictDataset): Dataset

    Returns:
        int: Dataset length
    """
    return len(dataset)


@dict_only_ds
def get_column_from_ds(dataset: DictDataset, column_name: str) -> np.ndarray:
    """Get a column from a DictDataset

    !!! note
        This function can be a bit time consuming since it needs to iterate
        over the whole dataset.

    Args:
        dataset (DictDataset): Dataset to get the column from
        column_name (str): Column value to get

    Returns:
        np.ndarray: Column values for dataset
    """

    columns = dataset.map(lambda x: x[column_name])
    columns = np.stack([f.numpy() for f in columns])
    return columns


def test_get_item_length():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    length = TorchDataHandler.get_item_length(data)
    assert length == 2


def test_instanciate_torch_datahandler():
    if os.environ["DL_LIB"] == "torch":
        handler = load_data_handler()
    else:
        handler = load_data_handler(backend="torch")
    assert isinstance(handler, TorchDataHandler)


def test_get_column_elements_shape():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    shape = TorchDataHandler.get_column_elements_shape(data, 0)
    assert shape == input_shape


def test_get_dataset_length():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    cardinality = TorchDataHandler.get_dataset_length(data)
    assert cardinality == samples


def test_get_input_from_dataset_item():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    tensor = TorchDataHandler.get_input_from_dataset_item(data[0])
    assert tensor.shape == (32, 32, 3)


@pytest.mark.parametrize(
    "dataset_name, train",
    [
        ("MNIST", True),
        ("MNIST", False),
    ],
)
def test_load_torchvision(dataset_name, train, erase_after_test=True):
    DATASET_INFOS = {
        "MNIST": {
            "img_shape": (1, 28, 28),
            "num_samples": {"train": 60000, "test": 10000},
        },
    }
    ds_infos = DATASET_INFOS[dataset_name]
    split = ["test", "train"][int(train)]

    handler = TorchDataHandler()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # define dataset
        dataset = handler.load_dataset(
            dataset_name, load_kwargs=dict(root=tmpdirname, train=train, download=True)
        )

        # dummy item
        dummy_item = dataset[0]
        dummy_columns = list(dummy_item.keys())
        dummy_shapes = {
            k: v.shape for k, v in zip(dummy_item.keys(), dummy_item.values())
        }

        # check columns
        assert dataset.column_names == dummy_columns == ["input", "label"]

        # check output shape
        assert (
            handler.get_columns_shapes(dataset)
            == dummy_shapes
            == {"input": torch.Size(ds_infos["img_shape"]), "label": torch.Size([])}
        )

        # check len of dataset
        assert len(dataset) == ds_infos["num_samples"][split]


@pytest.mark.parametrize(
    "dataset_name, split",
    [
        ("mnist", "train"),
        ("mnist", "test"),
    ],
)
def test_load_huggingface(dataset_name, split, erase_after_test=True):
    ds_infos = {
        "img_shape": (1, 28, 28),
        "num_samples": {"train": 60000, "test": 10000},
    }

    handler = TorchDataHandler()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # define dataset

        def transform(examples):
            examples["image"] = [
                torchvision.transforms.PILToTensor()(img) for img in examples["image"]
            ]
            return examples

        dataset = handler.load_dataset(
            dataset_name,
            load_kwargs=dict(cache_dir=tmpdirname, split=split, transform=transform),
            hub="huggingface",
        )

        # dummy item
        dummy_item = dataset[0]
        dummy_columns = list(dummy_item.keys())
        dummy_shapes = {
            k: v.shape for k, v in zip(dummy_item.keys(), dummy_item.values())
        }

        # check columns
        assert dataset.column_names == dummy_columns == ["image", "label"]

        # check output shape
        assert (
            handler.get_columns_shapes(dataset)
            == dummy_shapes
            == {"image": torch.Size(ds_infos["img_shape"]), "label": torch.Size([])}
        )

        # check len of dataset
        assert len(dataset) == ds_infos["num_samples"][split]


@pytest.mark.parametrize(
    "x_shape, num_samples, num_labels, one_hot",
    [
        ((3, 32, 32), 100, 10, True),
        ((1, 16, 16), 200, 2, False),
        ((64,), 1000, 20, False),
    ],
)
def test_load_arrays_and_custom(x_shape, num_labels, num_samples, one_hot):
    handler = TorchDataHandler()

    # === define datasets ===
    # tuple / dict numpy
    tuple_np = generate_data(
        x_shape=x_shape, num_labels=num_labels, samples=num_samples, one_hot=one_hot
    )
    dict_np = {"input": tuple_np[0], "label": tuple_np[1]}

    # tuple / dict torch
    tuple_torch = (torch.Tensor(tuple_np[0]), torch.Tensor(tuple_np[1]))
    dict_torch = {"input": tuple_torch[0], "label": tuple_torch[1]}

    # custom dataset (TensorDataset)
    tensor_ds_torch = generate_data_torch(
        x_shape=x_shape, num_labels=num_labels, samples=num_samples, one_hot=one_hot
    )

    # === load datasets ===
    for dataset_id in [tuple_np, dict_np, tuple_torch, dict_torch, tensor_ds_torch]:
        ds = handler.load_dataset(dataset_id, columns=["key_a", "key_b"])

        # check registered columns, shapes
        output_columns = ds.column_names
        output_shapes = handler.get_columns_shapes(ds)
        assert output_columns == ["key_a", "key_b"]
        assert output_shapes == {
            "key_a": torch.Size(x_shape),
            "key_b": torch.Size([num_labels] if one_hot else []),
        }
        # check item columns, shapes
        dummy_item = ds[0]
        assert list(dummy_item.keys()) == output_columns
        assert {
            k: v.shape for k, v in zip(dummy_item.keys(), dummy_item.values())
        } == output_shapes


@pytest.mark.parametrize(
    "x_shape, num_samples, num_labels, one_hot",
    [
        ((3, 32, 32), 150, 10, True),
        ((1, 16, 16), 200, 2, False),
        ((64,), 1000, 20, False),
    ],
)
def test_data_handler_full_pipeline(x_shape, num_samples, num_labels, one_hot):
    handler = TorchDataHandler()

    # define and load dataset
    dataset_id = generate_data(
        x_shape=x_shape, num_labels=num_labels, samples=num_samples, one_hot=one_hot
    )
    dataset = handler.load_dataset(dataset_id, columns=["input", "label"])
    assert len(dataset) == num_samples
    assert handler.get_columns_shapes(dataset)["input"] == torch.Size(x_shape)
    assert handler.get_columns_shapes(dataset)["label"] == (
        torch.Size([num_labels]) if one_hot else torch.Size([])
    )

    # filter by label
    a_labels = list(range(num_labels // 2))
    b_labels = list(range(num_labels // 2, num_labels))
    dataset_a = handler.filter_by_value(dataset, "label", a_labels)
    num_samples_a = len(dataset_a)
    dataset_b = handler.filter_by_value(dataset, "label", b_labels)
    num_samples_b = len(dataset_b)
    assert num_samples == (num_samples_a + num_samples_b)

    # assign column, map, get column
    def map_fn_a(item):
        item["new_column"] -= 3
        return item

    def map_fn_b(item):
        item["new_column"] = item["new_column"] * 3 + 2
        return item

    dataset_a = assign_value_to_column(dataset_a, "new_column", 0)
    dataset_a = dataset_a.map(map_fn_a)
    columns_a = torch.Tensor(get_column_from_ds(dataset_a, "new_column"))
    assert torch.all(columns_a == torch.Tensor([-3] * num_samples_a))

    dataset_b = assign_value_to_column(dataset_b, "new_column", 1)
    dataset_b = dataset_b.map(map_fn_b)
    columns_b = torch.Tensor(get_column_from_ds(dataset_b, "new_column"))
    assert torch.all(columns_b == torch.Tensor([5] * num_samples_b))

    # prepare dataloader
    loader = handler.prepare(dataset_b, 64, shuffle=True)
    batch = next(iter(loader))
    assert batch[0].shape == torch.Size([64, *x_shape])
    assert batch[1].shape == (
        torch.Size([64, num_labels]) if one_hot else torch.Size([64])
    )
    assert batch[2].shape == torch.Size([64])


@pytest.mark.parametrize(
    "in_labels, out_labels, one_hot, expected_output",
    [
        ([1, 2], None, False, [100, 67, 33, 2, 1]),
        (None, [1, 2], True, [100, 33, 67, 1, 2]),
        ([1], [2], False, [100, 33, 34, 1, 1]),
    ],
    ids=[
        "[torch] Assign OOD labels by class with ID labels",
        "[torch] Assign OOD labels by class with OOD labels",
        "[torch] Assign OOD labels by class with ID and OOD labels",
    ],
)
def test_split_by_class(in_labels, out_labels, one_hot, expected_output):
    """Test the split_by_class method."""

    # generate data
    x_shape = (3, 32, 32)
    images, labels = generate_data(
        x_shape=x_shape,
        num_labels=3,
        samples=100,
        one_hot=one_hot,
    )

    if not one_hot:
        for i in range(100):
            if i < 33:
                labels[i] = 0
            if i >= 33 and i < 66:
                labels[i] = 1
            if i >= 66:
                labels[i] = 2
    else:
        for i in range(100):
            if i < 33:
                labels[i] = np.array([1, 0, 0])
            if i >= 33 and i < 66:
                labels[i] = np.array([0, 1, 0])
            if i >= 66:
                labels[i] = np.array([0, 0, 1])

    handler = TorchDataHandler()
    dataset = handler.load_dataset(dataset_id=(images, labels))

    in_dataset, out_dataset = handler.split_by_class(
        dataset=dataset,
        in_labels=in_labels,
        out_labels=out_labels,
    )

    len_ds = len(dataset)
    len_inds = len(in_dataset)
    len_outds = len(out_dataset)

    classes = get_column_from_ds(dataset, "label")
    classes = np.unique(classes, axis=0)

    classes_in = get_column_from_ds(in_dataset, "label")
    classes_in = np.unique(classes_in, axis=0)

    classes_out = get_column_from_ds(out_dataset, "label")
    classes_out = np.unique(classes_out, axis=0)

    assert len_ds == expected_output[0]
    assert len_inds == expected_output[1]
    assert len_outds == expected_output[2]
    assert len(classes_in) == expected_output[3]
    assert len(classes_out) == expected_output[4]


@pytest.mark.parametrize(
    "shuffle, expected_output",
    [
        (False, [2, (16, 10)]),
        (True, [2, (16, 10)]),
    ],
    ids=[
        "[torch] Prepare OODDataset for scoring",
        "[torch] Prepare OODDataset for scoring (with shuffle and augment_fn)",
    ],
)
def test_prepare(shuffle, expected_output):
    """Test the prepare method."""

    num_labels = 10
    batch_size = 16
    samples = 100

    x_shape = (3, 32, 32)

    handler = TorchDataHandler()
    dataset = handler.load_dataset(
        dataset_id=generate_data_torch(
            x_shape=x_shape,
            num_labels=num_labels,
            samples=samples,
            one_hot=True,
        )
    )

    def preprocess_fn(inputs):
        inputs["input"] /= 255
        return inputs

    def augment_fn_(inputs):
        inputs["input"] = torchvision.transforms.RandomHorizontalFlip()(inputs["input"])
        return inputs

    augment_fn = augment_fn_ if shuffle else None

    ds = handler.prepare(
        dataset,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        shuffle=shuffle,
        augment_fn=augment_fn,
    )

    batch1 = next(iter(ds))
    batch2 = next(iter(ds))

    assert len(batch1) == expected_output[0]
    if shuffle:
        assert torch.sum(batch1[0] - batch2[0]) != 0
    assert batch1[0].shape == torch.Size([batch_size, 3, 32, 32])
    assert torch.max(batch1[0]) <= 1
    assert torch.min(batch1[0]) >= 0
