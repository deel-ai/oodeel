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


def get_dataset_length(dataset: Dataset) -> int:
    """Number of items in a dataset

    Args:
        dataset (DictDataset): Dataset

    Returns:
        int: Dataset length
    """
    return len(dataset)


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


def test_get_item_length():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    length = TorchDataHandler.get_item_length(data)
    assert length == 2


def test_instanciate_tf_datahandler():
    handler = load_data_handler(backend="torch")
    assert isinstance(handler, TorchDataHandler)


def test_get_feature_shape():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    shape = TorchDataHandler.get_feature_shape(data, 0)
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
def test_instanciate_from_torchvision(dataset_name, train, erase_after_test=True):
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
        dummy_keys = list(dummy_item.keys())
        dummy_shapes = [v.shape for v in dummy_item.values()]

        # check keys
        assert dataset.output_keys == dummy_keys == ["input", "label"]

        # check output shape
        assert (
            dataset.output_shapes
            == dummy_shapes
            == [torch.Size(ds_infos["img_shape"]), torch.Size([])]
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
        ds = handler.load_dataset(dataset_id, keys=["key_a", "key_b"])

        # check registered keys, shapes
        output_keys = ds.output_keys
        output_shapes = ds.output_shapes
        assert output_keys == ["key_a", "key_b"]
        assert output_shapes == [
            torch.Size(x_shape),
            torch.Size([num_labels] if one_hot else []),
        ]
        # check item keys, shapes
        dummy_item = ds[0]
        assert list(dummy_item.keys()) == output_keys
        assert list(map(lambda x: x.shape, dummy_item.values())) == output_shapes


@pytest.mark.parametrize(
    "x_shape, num_samples, num_labels, one_hot",
    [
        ((3, 32, 32), 100, 10, True),
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
    dataset = handler.load_dataset(dataset_id, keys=["input", "label"])
    assert len(dataset) == num_samples
    assert dataset.output_shapes[0] == torch.Size(x_shape)
    assert dataset.output_shapes[1] == (
        torch.Size([num_labels]) if one_hot else torch.Size([])
    )

    # filter by label
    a_labels = list(range(num_labels // 2))
    b_labels = list(range(num_labels // 2, num_labels))
    dataset_a = handler.filter_by_feature_value(dataset, "label", a_labels)
    num_samples_a = len(dataset_a)
    dataset_b = handler.filter_by_feature_value(dataset, "label", b_labels)
    num_samples_b = len(dataset_b)
    assert num_samples == (num_samples_a + num_samples_b)

    # assign feature, map, get feature
    def map_fn_a(item):
        item["new_feature"] -= 3
        return item

    def map_fn_b(item):
        item["new_feature"] = item["new_feature"] * 3 + 2
        return item

    dataset_a = assign_feature_value(dataset_a, "new_feature", 0)
    dataset_a = dataset_a.map(map_fn_a)
    features_a = torch.Tensor(get_feature_from_ds(dataset_a, "new_feature"))
    assert torch.all(features_a == torch.Tensor([-3] * num_samples_a))

    dataset_b = assign_feature_value(dataset_b, "new_feature", 1)
    dataset_b = dataset_b.map(map_fn_b)
    features_b = torch.Tensor(get_feature_from_ds(dataset_b, "new_feature"))
    assert torch.all(features_b == torch.Tensor([5] * num_samples_b))

    # concatenate two sub datasets
    dataset_c = handler.merge(dataset_a, dataset_b)
    features_c = torch.Tensor(get_feature_from_ds(dataset_c, "new_feature"))
    assert torch.all(features_c == torch.cat([features_a, features_b]))

    # prepare dataloader
    loader = handler.prepare_for_training(dataset_c, 64, True)
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

    classes = get_feature_from_ds(dataset, "label")
    classes = np.unique(classes, axis=0)

    classes_in = get_feature_from_ds(in_dataset, "label")
    classes_in = np.unique(classes_in, axis=0)

    classes_out = get_feature_from_ds(out_dataset, "label")
    classes_out = np.unique(classes_out, axis=0)

    assert len_ds == expected_output[0]
    assert len_inds == expected_output[1]
    assert len_outds == expected_output[2]
    assert len(classes_in) == expected_output[3]
    assert len(classes_out) == expected_output[4]


@pytest.mark.parametrize(
    "shuffle, with_labels, expected_output",
    [
        (False, True, [2, (16, 10)]),
        (False, False, [1, (16,)]),
        (True, True, [2, (16, 10)]),
        (True, False, [1, (16,)]),
    ],
    ids=[
        "[torch] Prepare OODDataset for scoring with labels and ood labels",
        "[torch] Prepare OODDataset for scoring with only ood labels",
        "[torch] Prepare OODDataset for training (with shuffle and augment_fn) with "
        "labels and ood labels",
        "[torch] Prepare OODDataset for training (with shuffle and augment_fn) "
        "with only labels",
    ],
)
def test_prepare(shuffle, with_labels, expected_output):
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

    def preprocess_fn(*inputs):
        x = inputs[0] / 255
        return tuple([x] + list(inputs[1:]))

    def augment_fn_(*inputs):
        x = torchvision.transforms.RandomHorizontalFlip()(inputs[0])
        return tuple([x] + list(inputs[1:]))

    augment_fn = augment_fn_ if shuffle else None

    ds = handler.prepare(
        dataset,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        with_labels=with_labels,
        shuffle=shuffle,
        augment_fn=augment_fn,
    )

    batch1 = next(iter(ds))
    batch2 = next(iter(ds))

    assert len(batch1) == expected_output[0]
    if shuffle:
        assert torch.sum(batch1[0] - batch2[0]) != 0
    if with_labels:
        assert batch1[1].shape == torch.Size([batch_size, num_labels])
    assert batch1[0].shape == torch.Size([batch_size, 3, 32, 32])
    assert torch.max(batch1[0]) <= 1
    assert torch.min(batch1[0]) >= 0
