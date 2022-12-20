import tensorflow as tf
from oodeel.types import *
from oodeel.utils.tools import *
from tests import generate_model, generate_data_tfds, almost_equal


def test_dataset_nb_columns():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    length = dataset_nb_columns(data)
    assert length == 2


def test_dataset_image_shape():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    shape = dataset_image_shape(data)
    assert shape == input_shape


def test_dataset_label_shape():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    shape = dataset_label_shape(data)
    assert shape == (num_labels,)


def test_dataset_max_pixel():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    max_pixels = dataset_max_pixel(data)
    assert max_pixels == 1.0


def test_dataset_nb_labels():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )  # .batch(samples)

    nb_labels = dataset_nb_labels(data)
    assert nb_labels == num_labels


def test_dataset_cardinality():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    cardinality = dataset_cardinality(data)
    assert cardinality == samples


def test_dataset_get_columns():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    data_0 = dataset_get_columns(data, 0)
    length = dataset_nb_columns(data_0)
    assert length == 1

    data_0 = dataset_get_columns(data, [1])
    length = dataset_nb_columns(data_0)
    assert length == 1

    data_0 = dataset_get_columns(data, [0, 1])
    length = dataset_nb_columns(data_0)
    assert length == 2
