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
import tensorflow as tf

from oodeel.models.keras_feature_extractor import KerasFeatureExtractor
from oodeel.types import *
from tests import almost_equal
from tests import generate_data_tfds
from tests import generate_model


def test_predict():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    )  # .batch(samples)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    feature_extractor = KerasFeatureExtractor(model, output_layers_id=[])

    model_fe = KerasFeatureExtractor(model, output_layers_id=[])

    last_layer = KerasFeatureExtractor(model, output_layers_id=[], input_layer_id=-2)

    pred_model = model.predict(data.batch(samples))
    pred_feature_extractor = feature_extractor.predict(data)
    pred_model_fe = model_fe.predict(data)
    pred_last_layer = last_layer.predict(pred_feature_extractor[0])

    assert almost_equal(pred_model, pred_model_fe)
    assert almost_equal(pred_model, pred_last_layer)


def test_gradient_pred():
    """
    Test gradient_mpred
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )  # .batch(samples)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    feature_extractor = KerasFeatureExtractor(model, output_layers_id=[], batch_size=26)

    grad_feature_extractor = feature_extractor.gradient_pred(data)

    assert list(grad_feature_extractor.shape) == ([samples] + list(input_shape))


def test_gradient_true_pred():
    """
    Test gradient_mpred
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )  # .batch(samples)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    feature_extractor = KerasFeatureExtractor(model, output_layers_id=[], batch_size=26)

    grad_feature_extractor = feature_extractor.gradient_true_pred(data)

    assert list(grad_feature_extractor.shape) == ([samples] + list(input_shape))


def test_gradient_index():
    """
    Test gradient_index
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )  # .batch(samples)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    feature_extractor = KerasFeatureExtractor(model, output_layers_id=[], batch_size=26)

    grad_feature_extractor = feature_extractor.gradient_index(data, index=4)

    assert list(grad_feature_extractor.shape) == ([samples] + list(input_shape))
