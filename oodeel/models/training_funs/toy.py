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
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tqdm import tqdm

from ...datasets import TFDataHandler
from ...types import List
from ...types import Optional
from ...types import Union


def train_convnet_classifier_tf(
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
    """Loads a model from tensorflow.python.keras.applications.
    If the dataset is different from imagenet, trains on provided dataset.

    Args:
        train_data (tf.data.Dataset)
        input_shape (tuple, optional): If None, infered from train_data.
            Defaults to None.
        num_classes (int, optional): If None, infered from train_data. Defaults to None.
        is_prepared (bool, optional): If train_data is a pipeline already prepared
            for training (with batch, shufle, cache etc...). Defaults to False.
        batch_size (int, optional): Defaults to 128.
        epochs (int, optional): Defaults to 50.
        loss (str, optional): Defaults to
            "sparse_categorical_crossentropy".
        optimizer (str, optional): Defaults to "adam".
        learning_rate (float, optional): Defaults to 1e-3.
        metrics (List[str], optional): Validation metrics. Defaults to ["accuracy"].
        imagenet_pretrained (bool, optional): Load a model pretrained on imagenet or
            not. Defaults to False.
        validation_data (Optional[tf.data.Dataset], optional): Defaults to None.
        save_dir (Optional[str], optional): Directory to save the model.
            Defaults to None.

    Returns:
        tf.keras.Model: Trained model
    """
    # Prepare model

    assert (num_classes is not None) and (
        input_shape is not None
    ), "Please specify num_classes and input_shape"

    model = Sequential(
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

    n_samples = TFDataHandler.get_dataset_length(train_data)

    # Prepare data
    if not is_prepared:

        def _preprocess_fn(*inputs):
            x = inputs[0] / 255
            return tuple([x] + list(inputs[1:]))

        padding = 4
        image_size = input_shape[1]
        target_size = image_size + padding * 2
        nb_channels = input_shape[2]

        def _augment_fn(images, labels):
            images = tf.image.pad_to_bounding_box(
                images, padding, padding, target_size, target_size
            )
            images = tf.image.random_crop(images, (image_size, image_size, nb_channels))
            images = tf.image.random_flip_left_right(images)
            return images, labels

        train_data = (
            train_data.map(
                _preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .shuffle(n_samples)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        if validation_data is not None:
            validation_data = (
                validation_data.map(
                    _preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                .batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

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
                save_best_only=True,
            )
        )

    if len(model_checkpoint_callback) == 0:
        model_checkpoint_callback = None

    # Prepare learning rate scheduler and optimizer
    n_steps = n_samples * epochs
    values = list(learning_rate * np.array([1, 0.1, 0.01]))
    boundaries = list(np.round(n_steps * np.array([1 / 3, 2 / 3])).astype(int))

    # optimizer
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )

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


class ComplexNet(nn.Module):
    def __init__(self, input_shape, nb_channels=3):
        super().__init__()
        self.nb_channels = nb_channels
        self.input_shape = input_shape

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(nb_channels, 6, 5)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("conv2", nn.Conv2d(6, 16, 5)),
                    ("relu2", nn.ReLU()),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        fc_input_shape = self._calculate_fc_input_shape()

        self.fcs = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(fc_input_shape, 120)),
                    ("fc2", nn.Linear(120, 84)),
                    ("fc3", nn.Linear(84, 10)),
                ]
            )
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fcs(x)
        return x

    def _calculate_fc_input_shape(self):
        input_tensor = torch.ones(tuple([1] + list(self.input_shape)))
        x = self.feature_extractor(input_tensor)
        output_size = x.view(x.size(0), -1).size(1)

        return output_size


def train_convnet_classifier_torch(
    train_data: Union[torch.utils.data.DataLoader, tf.data.Dataset],
    input_shape: tuple = None,
    num_classes: int = None,
    is_prepared: bool = False,
    batch_size: int = 128,
    epochs: int = 50,
    loss: str = "CrossEntropyLoss",
    optimizer: str = "Adam",
    learning_rate: float = 1e-3,
    validation_data: Optional[tf.data.Dataset] = None,
    save_dir: Optional[str] = None,
    cuda_idx: int = 0,
) -> nn.Module:
    """
    Load a model from torchvision.models and train it on a tfds dataset.

    Args:
        train_data (tf.data.Dataset)
        model_name (str): must be a model from torchvision.models
        input_shape (tuple, optional): If None, infered from train_data.
            Defaults to None.
        num_classes (int, optional): If None, infered from train_data. Defaults to None.
        is_prepared (bool, optional): If train_data is a pipeline already prepared
            for training (with batch, shufle, cache etc...). Defaults to False.
        batch_size (int, optional): Defaults to 128.
        epochs (int, optional): Defaults to 50.
        loss (str, optional): Defaults to
            "CrossEntropyLoss".
        optimizer (str, optional): Defaults to "Adam".
        learning_rate (float, optional): Defaults to 1e-3.
        metrics (List[str], optional): Validation metrics. Defaults to ["accuracy"].
        imagenet_pretrained (bool, optional): Load a model pretrained on imagenet or
            not. Defaults to False.
        validation_data (Optional[tf.data.Dataset], optional): Defaults to None.
        save_dir (Optional[str], optional): Directory to save the model.
            Defaults to None.
        cuda_idx (int): idx of cuda device to use. Defaults to 0.

    Returns:
        trained model
    """
    # device
    device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else "cpu")

    # Prepare model
    # if input_shape is None:
    #   input_shape = dataset_image_shape(train_data)
    if num_classes is None:
        classes = train_data.map(lambda x, y: y).unique()
        num_classes = len(list(classes.as_numpy_iterator()))

    nb_channels = input_shape[0]
    model = ComplexNet(input_shape, nb_channels)
    model.to(device)

    # Prepare data
    if not is_prepared:
        assert isinstance(train_data, tf.data.Dataset)
        padding = 4
        image_size = input_shape[1]
        target_size = image_size + padding * 2

        def _augment_fn(images, labels):
            images = tf.image.pad_to_bounding_box(
                images, padding, padding, target_size, target_size
            )
            images = tf.image.random_crop(images, (nb_channels, image_size, image_size))
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

    # define optimizer and learning rate scheduler
    n_steps = len(train_data) * epochs
    boundaries = list(np.round(n_steps * np.array([1 / 3, 2 / 3])).astype(int))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=boundaries, gamma=0.1
    )

    # define loss
    criterion = getattr(nn, loss)()

    # train
    model = _train(
        model,
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=save_dir,
        device=device,
    )
    return model


def _train(
    model,
    train_data,
    validation_data,
    epochs,
    criterion,
    optimizer,
    lr_scheduler,
    save_dir,
    device,
):
    """Torch training loop over tfds dataset

    Args:
        model (_type_): _description_
        train_data (_type_): _description_
        validation_data (_type_): _description_
        epochs (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        lr_scheduler (_type_): _description_
        save_dir (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    best_val_acc = None
    for epoch in range(epochs):
        # train phase
        model.train()
        running_loss, running_acc = 0.0, 0.0
        with tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs} [Train]") as iterator:
            for i, (inputs, labels) in enumerate(iterator):
                # convert [inputs, labels] into torch tensors
                inputs = torch.Tensor(inputs.numpy()).to(device)
                labels = torch.Tensor(labels.numpy()).long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                acc = torch.mean((outputs.argmax(-1) == labels).float())
                running_loss += loss.item()
                running_acc += acc.item()
                if i % max(len(iterator) // 100, 1) == 0:
                    iterator.set_postfix(
                        {
                            "Loss": f"{running_loss / (i + 1):.3f}",
                            "Acc": f"{running_acc / (i + 1):.3f}",
                        }
                    )
        lr_scheduler.step()

        # validation phase
        if validation_data is not None:
            model.eval()
            running_loss, running_acc = 0.0, 0.0
            with tqdm(
                validation_data, desc=f"Epoch {epoch + 1}/{epochs} [Val]"
            ) as iterator:
                for i, (inputs, labels) in enumerate(iterator):
                    # convert [inputs, labels] into torch tensors
                    inputs = torch.Tensor(inputs.numpy()).to(device)
                    labels = torch.Tensor(labels.numpy()).long().to(device)

                    with torch.no_grad():
                        outputs = model(inputs)

                    running_loss += criterion(outputs, labels).item()
                    running_acc += torch.mean(
                        (outputs.argmax(-1) == labels).float()
                    ).item()
                    if i % max(len(iterator) // 100, 1) == 0:
                        iterator.set_postfix(
                            {
                                "Loss": f"{running_loss / (i + 1):.3f}",
                                "Acc": f"{running_acc / (i + 1):.3f}",
                            }
                        )
            val_acc = running_acc / (i + 1)
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model, os.path.join(save_dir, "best.pt"))

    torch.save(model, os.path.join(save_dir, "last.pt"))
    return model
