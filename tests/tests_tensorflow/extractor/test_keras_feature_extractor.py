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
from oodeel.extractor.keras_feature_extractor import KerasFeatureExtractor
from tests.tests_tensorflow import almost_equal
from tests.tests_tensorflow import generate_data_tf
from tests.tests_tensorflow import generate_model


def test_predict():
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    ).batch(samples // 2)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    feature_extractor = KerasFeatureExtractor(model, output_layers_id=[-3])

    model_fe = KerasFeatureExtractor(model, output_layers_id=[-1])

    last_layer = KerasFeatureExtractor(model, output_layers_id=[-1], input_layer_id=-2)

    pred_model = model.predict(data)
    pred_feature_extractor = feature_extractor.predict(data)
    pred_model_fe = model_fe.predict(data)
    # To obtain the exact same result, add these lines:
    # pred_feature_extractor = tf.data.Dataset.from_tensor_slices(
    #    pred_feature_extractor
    # ).batch(samples // 2)
    pred_last_layer = last_layer.predict(pred_feature_extractor)

    assert almost_equal(pred_model, pred_model_fe)
    assert almost_equal(pred_model, pred_last_layer)


def test_get_weights():
    input_shape = (32, 32, 3)
    num_labels = 10

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    model_fe = KerasFeatureExtractor(model, output_layers_id=[-1])
    W, b = model_fe.get_weights(-1)

    assert W.shape == (900, 10)
    assert b.shape == (10,)


def test_predict_with_labels():
    """Assert that FeatureExtractor.predict() correctly returns features and labels when
    return_labels=True.

    Multiple tests are performed:
    - dataset with labels or without labels
    - dataset with one-hot encoded or sparse labels
    - single tensor instead of a dataset
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    # Generate dataset with sparse labels, with one-hot labels and without labels
    data = generate_data_tf(
        x_shape=input_shape,
        num_labels=num_labels,
        samples=samples,
        one_hot=False,
    ).batch(samples // 3)

    data_one_hot = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    ).batch(samples // 3)

    data_wo_labels = data.map(lambda x, y: x)

    # Generate model and feature extractor
    model = generate_model(input_shape=input_shape, output_shape=num_labels)
    feature_extractor = KerasFeatureExtractor(model, output_layers_id=[-3])

    # Assert predict without labels returned
    out = feature_extractor.predict(data, return_labels=False)
    assert out.shape == (samples, 15, 15, 4)

    # Assert predict with labels returned
    out = feature_extractor.predict(data, return_labels=True)
    assert len(out) == 2
    assert out[0].shape == (samples, 15, 15, 4)
    assert out[1].shape == (samples,)

    # Assert predict with labels returned (labels one-hot)
    out = feature_extractor.predict(data_one_hot, return_labels=True)
    assert len(out) == 2
    assert out[0].shape == (samples, 15, 15, 4)
    assert out[1].shape == (samples,)

    # Assert predict with labels returned but no labels in dataset
    out = feature_extractor.predict(data_wo_labels, return_labels=True)
    assert len(out) == 2
    assert out[1] is None

    # Assert predict with no labels in dataset and return_labels=False
    out = feature_extractor.predict(data_wo_labels, return_labels=False)
    assert out.shape == (samples, 15, 15, 4)

    # Assert predict of a single tensor with return_labels=False and True
    for batch in data_wo_labels.take(1):
        pass
    out = feature_extractor.predict(batch, return_labels=False)
    assert out.shape == (33, 15, 15, 4)
    out = feature_extractor.predict(batch, return_labels=True)
    assert len(out) == 2
    assert out[1] is None

    # Assert predict of a tuple (tensor, labels) with return_labels=False and True
    for batch in data_one_hot.take(1):
        pass
    out = feature_extractor.predict(batch, return_labels=False)
    assert out.shape == (33, 15, 15, 4)
    out = feature_extractor.predict(batch, return_labels=True)
    assert len(out) == 2
    assert out[1].shape == (33,)
