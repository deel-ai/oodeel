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
import numpy as np
import pytest
import tensorflow as tf

from oodeel.datasets import OODDataset
from oodeel.utils import dataset_cardinality
from tests import generate_data
from tests import generate_data_tf


def test_instanciate_from_tfds():
    dataset = OODDataset(dataset_id="mnist", split="test")

    assert len(dataset.data) == 10000
    assert dataset.len_elem == 2
    assert len(dataset.data.element_spec) == 2


@pytest.mark.parametrize(
    "as_supervised, expected_output",
    [
        (False, [100, 2, 2]),
        (True, [100, 2, 2]),
    ],
    ids=[
        "Instanciate from tf data without ood labels",
        "Instanciate from np data without ood labels",
    ],
)
def test_instanciate_ood_dataset(as_supervised, expected_output):
    """Test the instanciation of OODDataset."""

    dataset_id = generate_data_tf(
        x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=as_supervised
    )
    dataset = OODDataset(dataset_id=dataset_id)

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert len(dataset.data.element_spec) == expected_output[2]


@pytest.mark.parametrize(
    "as_tf_datasets, expected_output",
    [
        (False, [200, 2, 3, 0.5]),
        (True, [200, 2, 3, 0.5]),
    ],
    ids=[
        "Concatenate two OODDatasets",
        "Concatenate a OODDataset a tf.data.Dataset",
    ],
)
def test_add_ood_data(as_tf_datasets, expected_output):
    """Test the concatenation of OODDataset."""

    dataset1 = OODDataset(
        dataset_id=generate_data_tf(
            x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
        )
    )

    if as_tf_datasets:
        dataset2 = generate_data(x_shape=(38, 38, 3), num_labels=10, samples=100)
    else:
        dataset2 = OODDataset(
            dataset_id=generate_data_tf(
                x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
            )
        )

    dataset = dataset1.add_out_data(dataset2, shape=(23, 23))
    ood_labels = dataset.get_ood_labels()

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert len(dataset.data.element_spec) == expected_output[2]
    assert np.mean(ood_labels) == expected_output[3]


@pytest.mark.parametrize(
    "in_labels, out_labels, one_hot, expected_output",
    [
        ([1, 2], None, False, [100, 67, 33, 2, 1]),
        (None, [1, 2], True, [100, 33, 67, 1, 2]),
        ([1], [2], False, [100, 33, 34, 1, 1]),
    ],
    ids=[
        "Assign OOD labels by class with ID labels",
        "Assign OOD labels by class with OOD labels",
        "Assign OOD labels by class with ID and OOD labels",
    ],
)
def test_assign_ood_labels_by_class(in_labels, out_labels, one_hot, expected_output):
    """Test the assign_ood_labels_by_class method."""

    images, labels = generate_data(
        x_shape=(32, 32, 3),
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

    dataset = OODDataset(
        dataset_id=(images, labels),
    )

    in_dataset, out_dataset = dataset.assign_ood_labels_by_class(
        in_labels=in_labels,
        out_labels=out_labels,
    )

    len_ds = dataset_cardinality(dataset.data)
    len_inds = dataset_cardinality(in_dataset.data)
    len_outds = dataset_cardinality(out_dataset.data)

    classes = dataset.data.map(lambda elem: elem["label"])
    classes = np.array(list(classes.as_numpy_iterator())[:-1])
    classes = np.unique(classes, axis=0)

    classes_in = in_dataset.data.map(lambda elem: elem["label"])
    classes_in = np.array(list(classes_in.as_numpy_iterator())[:-1])
    classes_in = np.unique(classes_in, axis=0)

    classes_out = out_dataset.data.map(lambda elem: elem["label"])
    classes_out = np.array(list(classes_out.as_numpy_iterator())[:-1])
    classes_out = np.unique(classes_out, axis=0)

    assert len_ds == expected_output[0]
    assert len_inds == expected_output[1]
    assert len_outds == expected_output[2]
    assert len(classes_in) == expected_output[3]
    assert len(classes_out) == expected_output[4]


@pytest.mark.parametrize(
    "shuffle, with_labels, with_ood_labels, expected_output",
    [
        (False, True, True, [3, (32, 10)]),
        (False, False, True, [2, (32,)]),
        (True, True, True, [3, (32, 10)]),
        (True, True, False, [2, (32, 10)]),
    ],
    ids=[
        "Prepare OODDataset for scoring with labels and ood labels",
        "Prepare OODDataset for scoring with only ood labels",
        "Prepare OODDataset for training (with shuffle and augment_fn) with labels "
        "and ood labels",
        "Prepare OODDataset for training (with shuffle and augment_fn) "
        "with only labels",
    ],
)
def test_prepare(shuffle, with_labels, with_ood_labels, expected_output):
    """Test the prepare method."""

    num_labels = 10
    batch_size = 32
    samples = 100

    dataset = OODDataset(
        dataset_id=generate_data_tf(
            x_shape=(32, 32, 3),
            num_labels=num_labels,
            samples=samples,
            as_supervised=False,
            one_hot=True,
        )
    )

    def preprocess_fn(*inputs):
        x = inputs[0] / 255
        return tuple([x] + list(inputs[1:]))

    if shuffle:

        def augment_fn(*inputs):
            x = tf.image.random_flip_left_right(inputs[0])
            return tuple([x] + list(inputs[1:]))

    else:
        augment_fn = None

    dataset2 = OODDataset(
        dataset_id=generate_data_tf(
            x_shape=(32, 32, 3),
            num_labels=num_labels,
            samples=samples,
            as_supervised=False,
            one_hot=True,
        )
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

    for elem in ds.take(1):
        tensor1 = elem

    for elem in ds.take(1):
        tensor2 = elem

    assert len(tensor1) == expected_output[0]
    if shuffle:
        assert np.sum(tensor1[0] - tensor2[0]) != 0
    assert tuple(tensor1[1].shape) == expected_output[1]
    if with_ood_labels and with_labels:
        assert tuple(tensor1[2].shape) == (32,)
    assert tensor1[0].shape == (batch_size, 32, 32, 3)
    assert tf.reduce_max(tensor1[0]) <= 1
    assert tf.reduce_min(tensor1[0]) >= 0
