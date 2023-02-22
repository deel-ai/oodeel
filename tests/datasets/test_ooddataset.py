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


@pytest.mark.parametrize(
    "kwargs_factory, expected_output",
    [
        (
            lambda: dict(dataset_id="mnist", is_ood=False, split="test"),
            [10000, 2, 3, 0],
        ),
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
                is_ood=False,
            ),
            [100, 2, 3, 0],
        ),
        (
            lambda: dict(
                dataset_id=generate_data(
                    x_shape=(32, 32, 3), num_labels=10, samples=100
                ),
                is_ood=False,
            ),
            [100, 2, 3, 0],
        ),
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
                is_ood=None,
            ),
            [100, 2, 2, None],
        ),
        (
            lambda: dict(
                dataset_id=generate_data(
                    x_shape=(32, 32, 3), num_labels=10, samples=100
                ),
                is_ood=None,
            ),
            [100, 2, 2, None],
        ),
    ],
    ids=[
        "Instanciate from tfds",
        "Instanciate from tf dataset as ID",
        "Instanciate from np dataset as ID",
        "Instanciate from tf data without ood labels",
        "Instanciate from np data without ood labels",
    ],
)
def test_instanciate_ood_dataset(kwargs_factory, expected_output):
    """Test the instanciation of OODDataset.
    The instanciation involves using:
    * self.assign_ood_label
    So these methods are considered tested as well
    """
    dataset = OODDataset(**kwargs_factory())

    if dataset.has_ood_labels():
        ood_labels = np.mean(dataset.ood_labels)
    else:
        ood_labels = None

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert len(dataset.data.element_spec) == expected_output[2]
    assert ood_labels == expected_output[3]


@pytest.mark.parametrize(
    "is_ood, as_tf_datasets, expected_output",
    [
        (False, False, [200, 2, 3, 0.5]),
        (None, False, [200, 2, 3, 0.5]),
        (True, True, [200, 2, 3, 1]),
    ],
    ids=[
        "Concatenate two Datasets with OOD labels",
        "Concatenate two Datasets without OOD labels",
        "Concatenate a Dataset with OOD labels with a tf.data.Dataset",
    ],
)
def test_concatenate_ood_dataset(is_ood, as_tf_datasets, expected_output):
    """Test the concatenation of OODDataset.
    The concatenation involves using:
    * self.assign_ood_label
    So these methods are considered tested as well
    """

    dataset1 = OODDataset(
        dataset_id=generate_data_tf(
            x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
        ),
        is_ood=is_ood,
    )

    if as_tf_datasets:
        dataset2 = generate_data(x_shape=(38, 38, 3), num_labels=10, samples=100)
    else:
        dataset2 = OODDataset(
            dataset_id=generate_data_tf(
                x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
            ),
            is_ood=True,
        )

    if dataset1.has_ood_labels():
        dataset = dataset1.concatenate(dataset2)
    else:
        dataset = dataset1.concatenate(dataset2, ood_as_id=True, shape=(23, 23))

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert len(dataset.data.element_spec) == expected_output[2]
    assert np.mean(dataset.ood_labels) == expected_output[3]


@pytest.mark.parametrize(
    "id_labels, ood_labels, one_hot, expected_output",
    [
        ([1, 2], None, False, [100, 67, 33, 2, 1, 3]),
        (None, [1, 2], True, [100, 33, 67, 1, 2, 3]),
        ([1], [2], False, [67, 33, 34, 1, 1, 2]),
    ],
    ids=[
        "Assign OOD labels by class with ID labels",
        "Assign OOD labels by class with OOD labels",
        "Assign OOD labels by class with ID and OOD labels",
    ],
)
def test_assign_ood_labels_by_class(id_labels, ood_labels, one_hot, expected_output):
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

    id_dataset, ood_dataset = dataset.assign_ood_labels_by_class(
        id_labels=id_labels, ood_labels=ood_labels, return_filtered_ds=True
    )

    len_ds = dataset_cardinality(dataset.data)
    len_idds = dataset_cardinality(id_dataset.data)
    len_oodds = dataset_cardinality(ood_dataset.data)

    classes = dataset.data.map(lambda elem: elem["label"])
    classes = np.array(list(classes.as_numpy_iterator())[:-1])
    classes = np.unique(classes, axis=0)

    classes_id = id_dataset.data.map(lambda elem: elem["label"])
    classes_id = np.array(list(classes_id.as_numpy_iterator())[:-1])
    classes_id = np.unique(classes_id, axis=0)

    classes_ood = ood_dataset.data.map(lambda elem: elem["label"])
    classes_ood = np.array(list(classes_ood.as_numpy_iterator())[:-1])
    classes_ood = np.unique(classes_ood, axis=0)

    assert len_ds == expected_output[0]
    assert len_idds == expected_output[1]
    assert len_oodds == expected_output[2]
    assert len(classes_id) == expected_output[3]
    assert len(classes_ood) == expected_output[4]
    assert len(classes) == expected_output[5]
    assert np.mean(dataset.ood_labels) == expected_output[2] / expected_output[0]
    print(expected_output)


@pytest.mark.parametrize(
    "shuffle, with_labels, as_supervised, with_ood_labels, expected_output",
    [
        (False, True, False, True, [3, (32,)]),
        (False, False, False, True, [2, (32,)]),
        (True, True, True, True, [3, (32, 10)]),
        (True, True, True, False, [2, (32, 10)]),
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
def test_prepare(shuffle, with_labels, as_supervised, with_ood_labels, expected_output):
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
            one_hot=False,
        )
    )

    if as_supervised and with_ood_labels:

        def preprocess_fn(x, y, z):
            x = x / 255
            return x, y, z

        def augment_fn(x, y, z):
            x = tf.image.random_flip_left_right(x)
            return x, y, z

    elif as_supervised and not with_ood_labels:

        def preprocess_fn(x, y):
            x = x / 255
            return x, y

        def augment_fn(x, y):
            x = tf.image.random_flip_left_right(x)
            return x, y

    elif not as_supervised:

        def preprocess_fn(elem):
            elem["input"] = elem["input"] / 255
            if "label" in elem.keys():
                elem["label"] = tf.one_hot(elem["label"], num_labels)
            return elem

        augment_fn = None

    ds = dataset.prepare(
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        as_supervised=as_supervised,
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

    if not as_supervised:
        assert tf.reduce_sum(tensor1["input"] - tensor2["input"]) == 0
        assert tuple(tensor1["ood_label"].shape) == expected_output[1]
        if "label" in tensor1.keys():
            assert tuple(tensor1["label"].shape) == (32, 10)
        assert tensor1["input"].shape == (batch_size, 32, 32, 3)
        assert tf.reduce_max(tensor1["input"]) <= 1
        assert tf.reduce_min(tensor1["input"]) >= 0
    else:
        assert tf.reduce_sum(tensor1[0] - tensor2[0]) != 0
        assert tuple(tensor1[1].shape) == (32,)
        assert tuple(tensor1[-1].shape) == (32,)
        assert tensor1[0].shape == (batch_size, 32, 32, 3)
        assert tf.reduce_max(tensor1[0]) <= 1
        assert tf.reduce_min(tensor1[0]) >= 0
