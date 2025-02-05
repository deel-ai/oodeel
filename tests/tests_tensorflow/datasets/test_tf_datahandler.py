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
import tensorflow as tf

from oodeel.datasets.data_handler import load_data_handler
from oodeel.datasets.tf_data_handler import TFDataHandler
from tests.tests_tensorflow import generate_data
from tests.tests_tensorflow import generate_data_tf


def assign_feature_value(
    dataset: tf.data.Dataset, feature_key: str, value: int
) -> tf.data.Dataset:
    """Assign a value to a feature for every sample in a tf.data.Dataset

    Args:
        dataset (tf.data.Dataset): tf.data.Dataset to assign the value to
        feature_key (str): Feature to assign the value to
        value (int): Value to assign

    Returns:
        tf.data.Dataset
    """
    assert isinstance(dataset.element_spec, dict), "dataset elements must be dicts"

    def assign_value_to_feature(x):
        x[feature_key] = value
        return x

    dataset = dataset.map(assign_value_to_feature)
    return dataset


def get_dataset_length(dataset: tf.data.Dataset) -> int:
    """Get the length of a dataset. Try to access it with len(), and if not
    available, with a reduce op.

    Args:
        dataset (tf.data.Dataset): Dataset to process

    Returns:
        int: dataset length
    """
    try:
        return len(dataset)
    except TypeError:
        cardinality = dataset.reduce(0, lambda x, _: x + 1)
        return int(cardinality)


def get_feature_from_ds(dataset: tf.data.Dataset, feature_key: str) -> np.ndarray:
    """Get a feature from a tf.data.Dataset

    !!! note
        This function can be a bit time consuming since it needs to iterate
        over the whole dataset.

    Args:
        dataset (tf.data.Dataset): tf.data.Dataset to get the feature from
        feature_key (str): Feature value to get

    Returns:
        np.ndarray: Feature values for dataset
    """
    features = dataset.map(lambda x: x[feature_key])
    features = list(features.as_numpy_iterator())
    features = np.array(features)
    return features


def test_instanciate_tf_datahandler():
    handler = load_data_handler(backend="tensorflow")
    assert isinstance(handler, TFDataHandler)


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
    num_samples_a = get_dataset_length(dataset_a)
    dataset_b = handler.filter_by_feature_value(dataset, "label", b_labels)
    num_samples_b = get_dataset_length(dataset_b)
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
    features_a = tf.convert_to_tensor(get_feature_from_ds(dataset_a, "new_feature"))
    assert tf.reduce_all(features_a == tf.convert_to_tensor([-3] * num_samples_a))

    dataset_b = assign_feature_value(dataset_b, "new_feature", 1)
    dataset_b = dataset_b.map(map_fn_b)
    features_b = tf.convert_to_tensor(get_feature_from_ds(dataset_b, "new_feature"))
    assert tf.reduce_all(features_b == tf.convert_to_tensor([5] * num_samples_b))

    # concatenate two sub datasets
    dataset_c = handler.merge(dataset_a, dataset_b)
    features_c = tf.convert_to_tensor(get_feature_from_ds(dataset_c, "new_feature"))
    assert tf.reduce_all(features_c == tf.concat([features_a, features_b], axis=0))

    # prepare dataloader
    loader = handler.prepare(dataset_c, 64, shuffle=True)
    batch = next(iter(loader))
    assert batch[0].shape == tf.TensorShape([64, *x_shape])
    assert batch[1].shape == (
        tf.TensorShape([64, num_labels]) if one_hot else tf.TensorShape([64])
    )
    assert batch[2].shape == tf.TensorShape([64])


def test_instanciate_from_tfds():
    handler = TFDataHandler()
    dataset = handler.load_dataset(dataset_id="mnist", load_kwargs={"split": "test"})

    assert len(dataset) == 10000
    assert len(dataset.element_spec) == 2


@pytest.mark.parametrize(
    "as_supervised, expected_output",
    [
        (
            False,
            [
                100,
                2,
            ],
        ),
        (True, [100, 2]),
    ],
    ids=[
        "Instanciate from tf data without supervision",
        "Instanciate from np data with supervision",
    ],
)
def test_instanciate_ood_dataset(as_supervised, expected_output):
    """Test the instanciation of OODDataset."""

    dataset_id = generate_data_tf(
        x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=as_supervised
    )

    handler = TFDataHandler()
    dataset = handler.load_dataset(dataset_id=dataset_id)

    item_len = len(dataset.element_spec)

    assert len(dataset) == expected_output[0]
    assert item_len == expected_output[1]


@pytest.mark.parametrize(
    "in_labels, out_labels, one_hot, expected_output",
    [
        ([1, 2], None, False, [100, 67, 33, 2, 1]),
        (None, [1, 2], True, [100, 33, 67, 1, 2]),
        ([1], [2], False, [100, 33, 34, 1, 1]),
    ],
    ids=[
        "[tf] Assign OOD labels by class with ID labels",
        "[tf] Assign OOD labels by class with OOD labels",
        "[tf] Assign OOD labels by class with ID and OOD labels",
    ],
)
def test_split_by_class(in_labels, out_labels, one_hot, expected_output):
    """Test the split_by_class method."""

    # generate data
    x_shape = (32, 32, 3)
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

    handler = TFDataHandler()
    dataset = handler.load_dataset(dataset_id=(images, labels))

    in_dataset, out_dataset = handler.split_by_class(
        dataset=dataset,
        in_labels=in_labels,
        out_labels=out_labels,
    )

    len_ds = get_dataset_length(dataset)
    len_inds = get_dataset_length(in_dataset)
    len_outds = get_dataset_length(out_dataset)

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
    "shuffle, expected_output",
    [
        (False, [2, (16, 10)]),
        (True, [2, (16, 10)]),
    ],
    ids=[
        "[tf] Prepare dataset for scoring",
        "[tf] Prepare dataset for scoring (with shuffle and augment_fn)",
    ],
)
def test_prepare(shuffle, expected_output):
    """Test the prepare method."""

    num_labels = 10
    batch_size = 16
    samples = 100

    x_shape = (32, 32, 3)

    handler = TFDataHandler()
    dataset = handler.load_dataset(
        dataset_id=generate_data_tf(
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
        inputs["input"] = tf.image.random_flip_left_right(inputs["input"])
        return inputs

    augment_fn = augment_fn_ if shuffle else None

    ds = handler.prepare(
        dataset,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        shuffle=shuffle,
        augment_fn=augment_fn,
    )

    tensor1 = next(iter(ds.take(1)))
    tensor2 = next(iter(ds.take(1)))

    assert len(tensor1) == expected_output[0]
    if shuffle:
        assert np.sum(tensor1[0] - tensor2[0]) != 0
    assert tensor1[0].shape == (batch_size, 32, 32, 3)
    assert tf.reduce_max(tensor1[0]) <= 1
    assert tf.reduce_min(tensor1[0]) >= 0
