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
from typing import Optional

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

from ...utils import dataset_image_shape


def run_tf_on_cpu():
    """Run tensorflow on cpu device.
    It prevents tensorflow from allocating the totality of the GPU so that
    some VRAM remain free to train torch models.

    /!\\ This function needs to be called BEFORE loading the tfds dataset!
    """
    tf.config.set_visible_devices([], "GPU")


def train_torch_model(
    train_data: tf.data.Dataset,
    model_name: str = "resnet18",
    batch_size: int = 128,
    epochs: int = 50,
    loss: str = "CrossEntropyLoss",
    optimizer: str = "Adam",
    learning_rate: float = 1e-3,
    imagenet_pretrained: bool = False,
    validation_data: Optional[tf.data.Dataset] = None,
    save_dir: Optional[str] = None,
    cuda_idx: int = 0,
) -> nn.Module:
    """
    Load a model from torchvision.models and train it on a tfds dataset.

    Args:
        train_data: _description_
        model_name: _description_
        batch_size: _description_. Defaults to 128.
        epochs: _description_. Defaults to 50.
        loss: _description_. Defaults to "crossentropy".
        optimizer: _description_. Defaults to "adam".
        learning_rate: _description_. Defaults to 1e-3.
        metrics: _description_. Defaults to ["accuracy"].
        imagenet_pretrained: _description_. Defaults to False.
        validation_data: _description_. Defaults to None.
        cuda_idx: _description_. Defaults to 0.

    Returns:
        trained model
    """
    # device
    device = torch.device(f"cuda:{cuda_idx}" if cuda_idx is not None else "cpu")

    # Prepare model
    input_shape = dataset_image_shape(train_data)
    classes = train_data.map(lambda x, y: y).unique()
    num_classes = len(list(classes.as_numpy_iterator()))

    model = getattr(torchvision.models, model_name)(
        num_classes=num_classes, pretrained=imagenet_pretrained
    ).to(device)

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
                inputs = torch.Tensor(inputs.numpy()).permute(0, 3, 1, 2).to(device)
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
                    inputs = torch.Tensor(inputs.numpy()).permute(0, 3, 1, 2).to(device)
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
