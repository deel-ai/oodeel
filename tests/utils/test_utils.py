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
from oodeel.utils import is_from


def test_is_from():
    # === torch model / tensor ===
    import torch
    import torch.nn as nn

    torch_model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(32, 16, 3, 1, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 32 * 16, 10),
    )
    assert is_from(torch_model, "torch")

    torch_tensor = torch.randn((3, 32, 32))
    assert is_from(torch_tensor, "torch")

    # === keras model / tensor ===
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    keras_model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(10),
        ]
    )
    assert is_from(keras_model, "keras")

    tf_tensor = tf.random.normal((32, 32, 3))
    assert is_from(tf_tensor, "tensorflow")
