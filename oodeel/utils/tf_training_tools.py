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
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

from ..datasets.tf_data_handler import TFDataHandler
from ..types import List
from ..types import Optional
from ..types import Union


def get_toy_mlp(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """Basic keras MLP classifier for toy datasets.

    Args:
        input_shape (tuple): Input data shape.
        num_classes (int): Number of classes for the classification task.

    Returns:
        tf.keras.Model: model
    """
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dense(150, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def get_toy_keras_convnet(num_classes: int) -> tf.keras.Model:
    """Basic keras convolutional classifier for toy datasets.

    Args:
        num_classes (int): Number of classes for the classification task.

    Returns:
        tf.keras.Model: model
    """
    return Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )


def train_tf_model(
    train_data: tf.data.Dataset,
    model: Union[tf.keras.Model, str],
    input_shape: tuple,
    num_classes: int,
    batch_size: int = 128,
    epochs: int = 50,
    loss: str = "sparse_categorical_crossentropy",
    optimizer: str = "adam",
    lr_scheduler: Optional[str] = None,
    learning_rate: float = 1e-3,
    metrics: List[str] = ["accuracy"],
    imagenet_pretrained: bool = False,
    validation_data: Optional[tf.data.Dataset] = None,
    save_dir: Optional[str] = None,
    save_best_only: bool = True,
) -> tf.keras.Model:
    """Loads a model from tensorflow.python.keras.applications.
    If the dataset is different from imagenet, trains on provided dataset.

    Args:
        train_data (tf.data.Dataset): training dataset.
        model (Union[tf.keras.Model, str]): if a string is provided, must be a model
            from tf.keras.applications or "toy_convnet" or "toy_mlp"
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        batch_size (int, optional): Defaults to 128.
        epochs (int, optional): Defaults to 50.
        loss (str, optional): Defaults to "sparse_categorical_crossentropy".
        optimizer (str, optional): Defaults to "adam".
        lr_scheduler (str, optional): ("cosine" | "steps" | None). Defaults to None.
        learning_rate (float, optional): Defaults to 1e-3.
        metrics (List[str], optional): Validation metrics. Defaults to ["accuracy"].
        imagenet_pretrained (bool, optional): Load a model pretrained on imagenet or
            not. Defaults to False.
        validation_data (Optional[tf.data.Dataset], optional): Defaults to None.
        save_dir (Optional[str], optional): Directory to save the model.
            Defaults to None.
        save_best_only (bool): If False, saved model will be the last one. Defaults to
            True.

    Returns:
        tf.keras.Model: Trained model
    """
    # get data infos from dataset
    if isinstance(train_data.element_spec, dict):
        input_id = "image"
        label_id = "label"
    else:
        input_id = 0
        label_id = -1
    if input_shape is None:
        input_shape = TFDataHandler.get_feature_shape(train_data, input_id)
    if num_classes is None:
        classes = TFDataHandler.get_feature(train_data, label_id).unique()
        num_classes = len(list(classes.as_numpy_iterator()))

    # prepare model
    if isinstance(model, tf.keras.Model):
        pass
    elif isinstance(model, str):
        if model == "toy_convnet":
            model = get_toy_keras_convnet(num_classes)
        elif model == "toy_mlp":
            model = get_toy_mlp(input_shape, num_classes)
        else:
            weights = "imagenet" if imagenet_pretrained else None
            backbone = getattr(tf.keras.applications, model)(
                include_top=False, weights=weights, input_shape=input_shape
            )

            features = tf.keras.layers.Flatten()(backbone.layers[-1].output)
            output = tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            )(features)
            model = tf.keras.Model(backbone.layers[0].input, output)

    n_samples = TFDataHandler.get_dataset_length(train_data)

    # Prepare callbacks
    model_checkpoint_callback = []

    if save_dir is not None:
        checkpoint_filepath = save_dir
        model_checkpoint_callback.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                save_best_only=save_best_only,
            )
        )

    if len(model_checkpoint_callback) == 0:
        model_checkpoint_callback = None

    # optimizer
    decay_steps = int(epochs * n_samples / batch_size)
    if lr_scheduler == "cosine":
        learning_rate_fn = tf.keras.experimental.CosineDecay(
            learning_rate, decay_steps=decay_steps
        )
    elif lr_scheduler == "steps":
        values = list(learning_rate * np.array([1, 0.1, 0.01]))
        boundaries = list(np.round(decay_steps * np.array([1 / 3, 2 / 3])).astype(int))
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values
        )
    else:
        learning_rate_fn = learning_rate

    config = {
        "class_name": optimizer,
        "config": {
            "learning_rate": learning_rate_fn,
        },
    }

    if optimizer == "SGD":
        config["config"]["momentum"] = 0.9
        config["config"]["decay"] = 5e-4

    keras_optimizer = tf.keras.optimizers.get(config)

    model.compile(loss=loss, optimizer=keras_optimizer, metrics=metrics)

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=model_checkpoint_callback,
    )

    if save_dir is not None:
        model.load_weights(save_dir)
        model.save(save_dir)
    return model
