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
import tensorflow as tf

from oodeel.types import *
from oodeel.utils.tools import *
from tests import almost_equal
from tests import generate_data_tfds
from tests import generate_model


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
