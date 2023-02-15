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
from typing import List
from typing import Optional

import numpy as np
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from keras.layers import Dense
from keras.layers import Flatten

from ...utils import dataset_image_shape


def train_keras_app(
    train_data: tf.data.Dataset,
    model_name: str,
    batch_size: int = 128,
    epochs: int = 50,
    loss: str = "sparse_categorical_crossentropy",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    metrics: List[str] = ["accuracy"],
    imagenet_pretrained: bool = False,
    validation_data: Optional[tf.data.Dataset] = None,
    save_dir: Optional[str] = None,
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

    # Prepare model
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

        if model_name != "resnet18":
            backbone = getattr(tf.keras.applications, model_name)(
                include_top=False, weights=None, input_shape=input_shape
            )

    if model_name != "resnet18":
        features = Flatten()(backbone.layers[-1].output)
        output = Dense(
            num_classes,
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
            activation="softmax",
        )(features)
        model = tf.keras.Model(backbone.layers[0].input, output)
    else:
        ResNet18, _ = Classifiers.get("resnet18")
        model = ResNet18(input_shape, classes=num_classes, weights=None)

    # Prepare data
    padding = 4
    image_size = input_shape[0]
    target_size = image_size + padding * 2

    def _augment_fn(images, labels):

        images = tf.image.pad_to_bounding_box(
            images, padding, padding, target_size, target_size
        )
        images = tf.image.random_crop(images, (image_size, image_size, 3))
        images = tf.image.random_flip_left_right(images)
        return images, labels

    n_samples = len(train_data)
    train_data = (
        train_data.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(n_samples)
        .batch(batch_size)
    )

    if validation_data is not None:
        validation_data = validation_data.batch(batch_size)

    # Prepare callbacks
    model_checkpoint_callback = []

    if save_dir is not None:
        checkpoint_filepath = save_dir
        model_checkpoint_callback.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )
        )

    if len(model_checkpoint_callback) == 0:
        model_checkpoint_callback = None

    # Prepare learning rate scheduler and optimizer
    n_steps = len(train_data) * epochs
    values = list(learning_rate * np.array([1, 0.1, 0.01]))
    boundaries = list(np.round(n_steps * np.array([1 / 3, 2 / 3])).astype(int))

    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
    config = {"class_name": optimizer, "config": {"learning_rate": lr_scheduler}}
    keras_optimizer = tf.keras.optimizers.get(config)

    model.compile(loss=loss, optimizer=keras_optimizer, metrics=metrics)

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=model_checkpoint_callback,
    )

    return model
