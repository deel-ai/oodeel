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
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow import keras

from ...types import *
from ...utils import dataset_cardinality
from ...utils import dataset_image_shape


def train_keras_app(
    train_data: tf.data.Dataset,
    model_name: str,
    batch_size: int = 128,
    epochs: int = 50,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    metrics: List[str] = ["accuracy"],
    imagenet_pretrained: bool = False,
    validation_data: Optional[tf.data.Dataset] = None,
) -> tf.keras.Model:
    """
    Loads a model from keras.applications.
    If the dataset is different from imagenet, trains on provided dataset.

    Args:
        train_data: _description_
        model_name: _description_
        batch_size: _description_. Defaults to 128.
        epochs: _description_. Defaults to 50.
        loss: _description_. Defaults to "categorical_crossentropy".
        optimizer: _description_. Defaults to "adam".
        learning_rate: _description_. Defaults to 1e-3.
        metrics: _description_. Defaults to ["accuracy"].
        imagenet_pretrained: _description_. Defaults to False.
        validation_data: _description_. Defaults to None.

    Returns:
        trained model
    """
    if imagenet_pretrained:
        input_shape = (224, 224, 3)
        backbone = getattr(tf.keras.applications, model_name)(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        num_classes = 1000
    else:
        input_shape = dataset_image_shape(train_data)
        classes = train_data.map(lambda x, y: y).unique()
        num_classes = len(list(classes.as_numpy_iterator()))

        backbone = getattr(tf.keras.applications, model_name)(
            include_top=False, weights=None, input_shape=input_shape
        )

    features = Flatten()(backbone.layers[-1].output)
    output = Dense(num_classes, activation="softmax")(features)
    model = tf.keras.Model(backbone.layers[0].input, output)

    train_data = train_data.map(lambda x, y: (x, tf.one_hot(y, num_classes))).batch(
        batch_size
    )

    if validation_data is not None:
        validation_data = validation_data.map(
            lambda x, y: (x, tf.one_hot(y, num_classes))
        ).batch(batch_size)

    n_steps = dataset_cardinality(train_data) * epochs
    values = list(learning_rate * np.array([1, 0.1, 0.01]))
    boundaries = list(np.round(n_steps * np.array([1 / 3, 2 / 3])).astype(int))

    #### TODO
    # Add preprocessing (data augmentation)

    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )

    config = {"class_name": optimizer, "config": {"learning_rate": lr_scheduler}}

    keras_optimizer = tf.keras.optimizers.get(config)

    model.compile(loss=loss, optimizer=keras_optimizer, metrics=metrics)

    model.fit(train_data, validation_data=validation_data, epochs=epochs)

    return model
