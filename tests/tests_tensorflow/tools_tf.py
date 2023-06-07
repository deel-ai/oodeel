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


def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    return np.mean(np.abs(arr1 - arr2)) < epsilon


def generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv2D(4, kernel_size=(2, 2), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_shape))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    return model


def generate_regression_model(features_shape, output_shape=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=features_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="relu"))
    model.add(tf.keras.layers.Dense(output_shape))
    model.compile(loss="mean_absolute_error", optimizer="sgd", metrics=["accuracy"])

    return model


def simplest_mlp(num_features, num_classes):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100, one_hot=True):
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    x /= np.max(x)
    if one_hot:
        y = np.eye(num_labels)[np.random.randint(0, num_labels, samples)]
    else:
        y = np.random.randint(0, num_labels, samples)

    return x, y


def generate_data_tf(
    x_shape=(32, 32, 3), num_labels=10, samples=100, one_hot=True, as_supervised=True
):
    x, y = generate_data(x_shape, num_labels, samples, one_hot)
    if as_supervised:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices({"input": x, "label": y})
    return dataset
