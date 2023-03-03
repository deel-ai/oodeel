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
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D

from ...types import List
from ...types import Optional
from ...utils import dataset_cardinality


def train_convnet_classifier(
    train_data: tf.data.Dataset,
    input_shape: tuple = None,
    num_classes: int = None,
    is_prepared: bool = False,
    batch_size: int = 128,
    epochs: int = 50,
    loss: str = "sparse_categorical_crossentropy",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    metrics: List[str] = ["accuracy"],
    validation_data: Optional[tf.data.Dataset] = None,
    save_dir: Optional[str] = None,
) -> tf.keras.Model:
    """
    Loads a model from tensorflow.python.keras.applications.
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

    assert (num_classes is not None) and (
        input_shape is not None
    ), "Please specify num_classes and input_shape"

    model = tf.keras.Sequential(
        [
            # keras.Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    # Prepare data
    if not is_prepared:
        padding = 4
        image_size = input_shape[0]
        nb_channels = input_shape[2]
        target_size = image_size + padding * 2

        def _augment_fn(images, labels):
            images = tf.image.pad_to_bounding_box(
                images, padding, padding, target_size, target_size
            )
            images = tf.image.random_crop(images, (image_size, image_size, nb_channels))
            images = tf.image.random_flip_left_right(images)
            return images, labels

        n_samples = len(train_data)
        train_data = (
            train_data.map(
                _augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
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
    n_steps = dataset_cardinality(train_data) * epochs
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
