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
import shutil

import pytest
import torch

from oodeel.datasets.torch_data_handler import TorchDataHandler
from tests import generate_data
from tests import generate_data_torch


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

    # temp dataset root
    temp_root = "./temp_dataset"
    os.makedirs(temp_root, exist_ok=True)

    handler = TorchDataHandler()

    # define dataset
    dataset = handler.load_dataset(
        dataset_name, load_kwargs=dict(root=temp_root, train=train, download=True)
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

    if erase_after_test:
        shutil.rmtree(temp_root, ignore_errors=True)


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

    dataset_a = handler.assign_feature_value(dataset_a, "new_feature", 0)
    dataset_a = dataset_a.map(map_fn_a)
    features_a = torch.Tensor(handler.get_feature_from_ds(dataset_a, "new_feature"))
    assert torch.all(features_a == torch.Tensor([-3] * num_samples_a))

    dataset_b = handler.assign_feature_value(dataset_b, "new_feature", 1)
    dataset_b = dataset_b.map(map_fn_b)
    features_b = torch.Tensor(handler.get_feature_from_ds(dataset_b, "new_feature"))
    assert torch.all(features_b == torch.Tensor([5] * num_samples_b))

    # concatenate two sub datasets
    dataset_c = handler.merge(dataset_a, dataset_b)
    features_c = torch.Tensor(handler.get_feature_from_ds(dataset_c, "new_feature"))
    assert torch.all(features_c == torch.cat([features_a, features_b]))

    # prepare dataloader
    loader = handler.prepare_for_training(dataset_c, 64, True)
    batch = next(iter(loader))
    assert batch[0].shape == torch.Size([64, *x_shape])
    assert batch[1].shape == (
        torch.Size([64, num_labels]) if one_hot else torch.Size([64])
    )
    assert batch[2].shape == torch.Size([64])
