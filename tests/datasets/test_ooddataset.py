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

import numpy as np
import pytest
import tensorflow as tf
import torch
import torchvision

from oodeel.datasets import OODDataset
from tests import generate_data
from tests import generate_data_tf
from tests import generate_data_torch


def test_instanciate_from_tfds():
    dataset = OODDataset(dataset_id="mnist", split="test")

    assert len(dataset.data) == 10000
    assert dataset.len_elem == 2
    assert len(dataset.data.element_spec) == 2


def test_instanciate_from_torchvision(erase_after_test=True):
    temp_root = "./temp_dataset"
    os.makedirs(temp_root, exist_ok=True)
    dataset = OODDataset(
        dataset_id="MNIST",
        split="test",
        backend="torch",
        load_kwargs=dict(root=temp_root, download=True),
    )

    assert len(dataset.data) == 10000
    assert dataset.len_elem == 2

    if erase_after_test:
        shutil.rmtree(temp_root, ignore_errors=True)


@pytest.mark.parametrize(
    "backend, as_supervised, expected_output",
    [
        ("tensorflow", False, [100, 2, 2]),
        ("tensorflow", True, [100, 2, 2]),
        ("torch", None, [100, 2, 2]),
    ],
    ids=[
        "Instanciate from tf data without supervision",
        "Instanciate from np data with supervision",
        "Instanciate from torch data",
    ],
)
def test_instanciate_ood_dataset(backend, as_supervised, expected_output):
    """Test the instanciation of OODDataset."""

    if backend == "tensorflow":
        dataset_id = generate_data_tf(
            x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=as_supervised
        )
    elif backend == "torch":
        dataset_id = generate_data_torch(
            x_shape=(3, 32, 32), num_labels=10, samples=100
        )

    dataset = OODDataset(dataset_id=dataset_id, backend=backend)

    item_len = (
        len(dataset.data.element_spec)
        if backend == "tensorflow"
        else len(dataset.data[0])
    )

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert item_len == expected_output[2]


@pytest.mark.parametrize(
    "backend, ds2_from_numpy, expected_output",
    [
        ("tensorflow", True, [200, 2, 3, 0.5]),
        ("tensorflow", False, [200, 2, 3, 0.5]),
        ("torch", True, [200, 2, 3, 0.5]),
        ("torch", False, [200, 2, 3, 0.5]),
    ],
    ids=[
        "[tf] Concatenate two OODDatasets",
        "[tf] Concatenate a OODDataset and a numpy dataset",
        "[torch] Concatenate two OODDatasets",
        "[torch] Concatenate a OODDataset and a numpy dataset",
    ],
)
def test_add_ood_data(backend, ds2_from_numpy, expected_output):
    """Test the concatenation of OODDataset."""

    generate_data_func = {"tensorflow": generate_data_tf, "torch": generate_data_torch}[
        backend
    ]
    x_shape = {"tensorflow": (32, 32, 3), "torch": (3, 32, 32)}[backend]

    dataset1 = OODDataset(
        dataset_id=generate_data_func(x_shape=x_shape, num_labels=10, samples=100),
        backend=backend,
    )

    if ds2_from_numpy:
        dataset2 = generate_data(x_shape=(38, 38, 3), num_labels=10, samples=100)
    else:
        dataset2 = OODDataset(
            dataset_id=generate_data_func(x_shape=x_shape, num_labels=10, samples=100),
            backend=backend,
        )

    dataset = dataset1.add_out_data(dataset2, shape=(23, 23))
    ood_labels = dataset.get_ood_labels()
    item_len = (
        len(dataset.data.element_spec)
        if backend == "tensorflow"
        else len(dataset.data[0])
    )
    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert item_len == expected_output[2]
    assert np.mean(ood_labels) == expected_output[3]


@pytest.mark.parametrize(
    "backend, in_labels, out_labels, one_hot, expected_output",
    [
        ("tensorflow", [1, 2], None, False, [100, 67, 33, 2, 1]),
        ("tensorflow", None, [1, 2], True, [100, 33, 67, 1, 2]),
        ("tensorflow", [1], [2], False, [100, 33, 34, 1, 1]),
        ("torch", [1, 2], None, False, [100, 67, 33, 2, 1]),
        ("torch", None, [1, 2], True, [100, 33, 67, 1, 2]),
        ("torch", [1], [2], False, [100, 33, 34, 1, 1]),
    ],
    ids=[
        "[tf] Assign OOD labels by class with ID labels",
        "[tf] Assign OOD labels by class with OOD labels",
        "[tf] Assign OOD labels by class with ID and OOD labels",
        "[torch] Assign OOD labels by class with ID labels",
        "[torch] Assign OOD labels by class with OOD labels",
        "[torch] Assign OOD labels by class with ID and OOD labels",
    ],
)
def test_assign_ood_labels_by_class(
    backend, in_labels, out_labels, one_hot, expected_output
):
    """Test the assign_ood_labels_by_class method."""

    # generate data
    x_shape = (32, 32, 3) if backend == "tensorflow" else (3, 32, 32)
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

    dataset = OODDataset(dataset_id=(images, labels), backend=backend)

    in_dataset, out_dataset = dataset.assign_ood_labels_by_class(
        in_labels=in_labels,
        out_labels=out_labels,
    )

    len_ds = len(dataset)
    len_inds = len(in_dataset)
    len_outds = len(out_dataset)

    classes = dataset._data_handler.get_feature_from_ds(dataset.data, "label")
    classes = np.unique(classes, axis=0)

    classes_in = dataset._data_handler.get_feature_from_ds(in_dataset.data, "label")
    classes_in = np.unique(classes_in, axis=0)

    classes_out = dataset._data_handler.get_feature_from_ds(out_dataset.data, "label")
    classes_out = np.unique(classes_out, axis=0)

    assert len_ds == expected_output[0]
    assert len_inds == expected_output[1]
    assert len_outds == expected_output[2]
    assert len(classes_in) == expected_output[3]
    assert len(classes_out) == expected_output[4]


@pytest.mark.parametrize(
    "backend, shuffle, with_labels, with_ood_labels, expected_output",
    [
        ("tensorflow", False, True, True, [3, (32, 10)]),
        ("tensorflow", False, False, True, [2, (32,)]),
        ("tensorflow", True, True, True, [3, (32, 10)]),
        ("tensorflow", True, True, False, [2, (32, 10)]),
        ("torch", False, True, True, [3, (32, 10)]),
        ("torch", False, False, True, [2, (32,)]),
        ("torch", True, True, True, [3, (32, 10)]),
        ("torch", True, True, False, [2, (32, 10)]),
    ],
    ids=[
        "[tf] Prepare OODDataset for scoring with labels and ood labels",
        "[tf] Prepare OODDataset for scoring with only ood labels",
        "[tf] Prepare OODDataset for training (with shuffle and augment_fn) with labels "
        "and ood labels",
        "[tf] Prepare OODDataset for training (with shuffle and augment_fn) "
        "with only labels",
        "[torch] Prepare OODDataset for scoring with labels and ood labels",
        "[torch] Prepare OODDataset for scoring with only ood labels",
        "[torch] Prepare OODDataset for training (with shuffle and augment_fn) with "
        "labels and ood labels",
        "[torch] Prepare OODDataset for training (with shuffle and augment_fn) "
        "with only labels",
    ],
)
def test_prepare(backend, shuffle, with_labels, with_ood_labels, expected_output):
    """Test the prepare method."""

    num_labels = 10
    batch_size = 32
    samples = 100

    x_shape = (32, 32, 3) if backend == "tensorflow" else (3, 32, 32)
    generate_data_fn = (
        generate_data_tf if backend == "tensorflow" else generate_data_torch
    )

    dataset = OODDataset(
        dataset_id=generate_data_fn(
            x_shape=x_shape,
            num_labels=num_labels,
            samples=samples,
            one_hot=True,
        ),
        backend=backend,
    )

    if backend == "tensorflow":

        def preprocess_fn(*inputs):
            x = inputs[0] / 255
            return tuple([x] + list(inputs[1:]))

        def augment_fn(*inputs):
            x = tf.image.random_flip_left_right(inputs[0])
            return tuple([x] + list(inputs[1:]))

    else:

        def preprocess_fn(item_dict):
            item_dict["input"] /= 255.0
            return item_dict

        def augment_fn(item_dict):
            item_dict["input"] = torchvision.transforms.RandomHorizontalFlip()(
                item_dict["input"]
            )
            return item_dict

    if shuffle is False:
        augment_fn = None

    dataset2 = OODDataset(
        dataset_id=generate_data_fn(
            x_shape=x_shape,
            num_labels=num_labels,
            samples=samples,
            one_hot=True,
        ),
        backend=backend,
    )

    dataset = dataset.add_out_data(dataset2)

    ds = dataset.prepare(
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        with_labels=with_labels,
        with_ood_labels=with_ood_labels,
        shuffle=shuffle,
        augment_fn=augment_fn,
    )

    if backend == "tensorflow":
        tensor1 = next(iter(ds.take(1)))
        tensor2 = next(iter(ds.take(1)))

        assert len(tensor1) == expected_output[0]
        if shuffle:
            assert np.sum(tensor1[0] - tensor2[0]) != 0
        assert tuple(tensor1[1].shape) == expected_output[1]
        if with_ood_labels and with_labels:
            assert tuple(tensor1[2].shape) == (32,)
        assert tensor1[0].shape == (batch_size, 32, 32, 3)
        assert tf.reduce_max(tensor1[0]) <= 1
        assert tf.reduce_min(tensor1[0]) >= 0
    elif backend == "torch":
        batch1 = next(iter(ds))
        batch2 = next(iter(ds))

        assert len(batch1) == expected_output[0]
        if shuffle:
            assert torch.sum(batch1[0] - batch2[0]) != 0
        assert batch1[1].shape == torch.Size(expected_output[1])
        if with_ood_labels and with_labels:
            assert batch1[2].shape == torch.Size([32])
        assert batch1[0].shape == torch.Size([batch_size, 3, 32, 32])
        assert torch.max(batch1[0]) <= 1
        assert torch.min(batch1[0]) >= 0
