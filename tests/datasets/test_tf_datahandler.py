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

import pytest
import tensorflow as tf

from oodeel.datasets.tf_data_handler import TFDataHandler
from tests import generate_data
from tests import generate_data_tf


def test_get_item_length():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    length = TFDataHandler.get_item_length(data)
    assert length == 2


def test_get_feature_shape():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    shape = TFDataHandler.get_feature_shape(data, 0)
    assert shape == input_shape


def test_get_dataset_length():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    cardinality = TFDataHandler.get_dataset_length(data)
    assert cardinality == samples


def test_get_input_from_dataset_item():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    for datum in data.take(1):
        tensor = TFDataHandler.get_input_from_dataset_item(datum)
    assert tensor.shape == (32, 32, 3)


@pytest.mark.parametrize(
    "dataset_name, train",
    [
        ("mnist", True),
        ("mnist", False),
    ],
)
def test_load_tensorflow_datasets(dataset_name, train):
    DATASET_INFOS = {
        "mnist": {
            "img_shape": (28, 28, 1),
            "num_samples": {"train": 60000, "test": 10000},
        }
    }
    ds_infos = DATASET_INFOS[dataset_name]
    split = ["test", "train"][int(train)]

    handler = TFDataHandler()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # define dataset
        dataset = handler.load_dataset(
            dataset_name,
            load_kwargs=dict(data_dir=tmpdirname, split=split, download=True),
        )

        # dummy item
        for item in dataset.take(1):
            dummy_item = item
        dummy_keys = list(dummy_item.keys())
        dummy_shapes = [v.shape for v in dummy_item.values()]

        # check keys
        assert list(dataset.element_spec.keys()) == dummy_keys == ["image", "label"]

        # check output shape
        assert (
            [dataset.element_spec[key].shape for key in dataset.element_spec.keys()]
            == dummy_shapes
            == [tf.TensorShape(ds_infos["img_shape"]), tf.TensorShape([])]
        )

        # check len of dataset
        assert len(dataset) == ds_infos["num_samples"][split]


@pytest.mark.parametrize(
    "x_shape, num_samples, num_labels, one_hot",
    [
        ((32, 32, 3), 100, 10, True),
        ((16, 16, 1), 200, 2, False),
        ((64,), 1000, 20, False),
    ],
)
def test_load_arrays_and_custom(x_shape, num_labels, num_samples, one_hot):
    handler = TFDataHandler()

    # === define datasets ===
    # tuple / dict numpy
    tuple_np = generate_data(
        x_shape=x_shape, num_labels=num_labels, samples=num_samples, one_hot=one_hot
    )
    dict_np = {"input": tuple_np[0], "label": tuple_np[1]}

    # tuple / dict torch
    tuple_tf = (
        tf.convert_to_tensor(tuple_np[0]),
        tf.convert_to_tensor(tuple_np[1]),
    )
    dict_tf = {"input": tuple_tf[0], "label": tuple_tf[1]}

    # custom dataset (TensorDataset)
    tensor_ds_tf = generate_data_tf(
        x_shape=x_shape, num_labels=num_labels, samples=num_samples, one_hot=one_hot
    )

    # === load datasets ===
    for dataset_id in [tuple_np, dict_np, tuple_tf, dict_tf, tensor_ds_tf]:
        ds = handler.load_dataset(dataset_id, keys=["key_a", "key_b"])

        # check registered keys, shapes
        output_keys = list(ds.element_spec.keys())
        output_shapes = [ds.element_spec[key].shape for key in ds.element_spec.keys()]
        assert output_keys == ["key_a", "key_b"]
        assert output_shapes == [
            tf.TensorShape(x_shape),
            tf.TensorShape([num_labels] if one_hot else []),
        ]
        # check item keys, shapes
        for item in ds.take(1):
            dummy_item = item
        assert list(dummy_item.keys()) == output_keys
        assert list(map(lambda x: x.shape, dummy_item.values())) == output_shapes


@pytest.mark.parametrize(
    "x_shape, num_samples, num_labels, one_hot",
    [
        ((32, 32, 3), 100, 10, True),
        ((16, 16, 1), 200, 2, False),
        ((64,), 1000, 20, False),
    ],
)
def test_data_handler_full_pipeline(x_shape, num_samples, num_labels, one_hot):
    handler = TFDataHandler()

    # define and load dataset
    dataset_id = generate_data(
        x_shape=x_shape, num_labels=num_labels, samples=num_samples, one_hot=one_hot
    )
    dataset = handler.load_dataset(dataset_id, keys=["input", "label"])
    assert len(dataset) == num_samples
    assert dataset.element_spec["input"].shape == tf.TensorShape(x_shape)
    assert dataset.element_spec["label"].shape == (
        tf.TensorShape([num_labels]) if one_hot else tf.TensorShape([])
    )

    # filter by label
    a_labels = list(range(num_labels // 2))
    b_labels = list(range(num_labels // 2, num_labels))
    dataset_a = handler.filter_by_feature_value(dataset, "label", a_labels)
    num_samples_a = handler.get_dataset_length(dataset_a)
    dataset_b = handler.filter_by_feature_value(dataset, "label", b_labels)
    num_samples_b = handler.get_dataset_length(dataset_b)
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
    features_a = tf.convert_to_tensor(
        handler.get_feature_from_ds(dataset_a, "new_feature")
    )
    assert tf.reduce_all(features_a == tf.convert_to_tensor([-3] * num_samples_a))

    dataset_b = handler.assign_feature_value(dataset_b, "new_feature", 1)
    dataset_b = dataset_b.map(map_fn_b)
    features_b = tf.convert_to_tensor(
        handler.get_feature_from_ds(dataset_b, "new_feature")
    )
    assert tf.reduce_all(features_b == tf.convert_to_tensor([5] * num_samples_b))

    # concatenate two sub datasets
    dataset_c = handler.merge(dataset_a, dataset_b)
    features_c = tf.convert_to_tensor(
        handler.get_feature_from_ds(dataset_c, "new_feature")
    )
    assert tf.reduce_all(features_c == tf.concat([features_a, features_b], axis=0))

    # prepare dataloader
    loader = handler.prepare_for_training(dataset_c, 64, True)
    batch = next(iter(loader))
    assert batch[0].shape == tf.TensorShape([64, *x_shape])
    assert batch[1].shape == (
        tf.TensorShape([64, num_labels]) if one_hot else tf.TensorShape([64])
    )
    assert batch[2].shape == tf.TensorShape([64])
