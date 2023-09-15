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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..types import Optional
from ..types import Union


class ToyTorchMLP(nn.Sequential):
    """Basic torch MLP classifier for toy datasets.

    Args:
        input_shape (tuple): Input data shape.
        num_classes (int): Number of classes for the classification task.
    """

    def __init__(self, input_shape: tuple, num_classes: int):
        self.input_shape = input_shape

        # build toy mlp
        mlp_modules = OrderedDict(
            [
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(np.prod(input_shape), 300)),
                ("relu1", nn.ReLU()),
                ("dense2", nn.Linear(300, 150)),
                ("relu2", nn.ReLU()),
                ("fc1", nn.Linear(150, num_classes)),
            ]
        )
        super().__init__(mlp_modules)


class ToyTorchConvnet(nn.Sequential):
    """Basic torch convolutional classifier for toy datasets.

    Args:
        input_shape (tuple): Input data shape.
        num_classes (int): Number of classes for the classification task.
    """

    def __init__(self, input_shape: tuple, num_classes: int):
        self.input_shape = input_shape

        # features
        features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(input_shape[0], 32, 3)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("conv2", nn.Conv2d(32, 64, 3)),
                    ("relu2", nn.ReLU()),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        # fc head
        fc_input_shape = self._calculate_fc_input_shape(features)
        fcs = nn.Sequential(
            OrderedDict(
                [
                    ("dropout", nn.Dropout(0.5)),
                    ("fc1", nn.Linear(fc_input_shape, num_classes)),
                ]
            )
        )

        # Sequential class init
        super().__init__(
            OrderedDict([*features._modules.items(), *fcs._modules.items()])
        )

    def _calculate_fc_input_shape(self, features):
        """Get tensor shape after passing a features network."""
        input_tensor = torch.ones(tuple([1] + list(self.input_shape)))
        x = features(input_tensor)
        output_size = x.view(x.size(0), -1).size(1)
        return output_size


def train_torch_model(
    train_data: DataLoader,
    model: Union[nn.Module, str],
    num_classes: int,
    epochs: int = 50,
    loss: str = "CrossEntropyLoss",
    optimizer: str = "Adam",
    lr_scheduler: str = "cosine",
    learning_rate: float = 1e-3,
    imagenet_pretrained: bool = False,
    validation_data: Optional[DataLoader] = None,
    save_dir: Optional[str] = None,
    cuda_idx: int = 0,
) -> nn.Module:
    """
    Load a model (toy classifier or from torchvision.models) and train
    it over a torch dataloader.

    Args:
        train_data (DataLoader): train dataloader
        model (Union[nn.Module, str]): if a string is provided, must be a model from
            torchvision.models or "toy_convnet" or "toy_mlp.
        num_classes (int): Number of output classes.
        epochs (int, optional): Defaults to 50.
        loss (str, optional): Defaults to "CrossEntropyLoss".
        optimizer (str, optional): Defaults to "Adam".
        lr_scheduler (str, optional): ("cosine" | "steps" | None). Defaults to None.
        learning_rate (float, optional): Defaults to 1e-3.
        imagenet_pretrained (bool, optional): Load a model pretrained on imagenet or
            not. Defaults to False.
        validation_data (Optional[DataLoader], optional): Defaults to None.
        save_dir (Optional[str], optional): Directory to save the model.
            Defaults to None.
        cuda_idx (int): idx of cuda device to use. Defaults to 0.

    Returns:
        nn.Module: trained model
    """
    # device
    device = torch.device(
        f"cuda:{cuda_idx}"
        if torch.cuda.is_available() and cuda_idx is not None
        else "cpu"
    )

    # Prepare model
    if isinstance(model, nn.Module):
        model = model.to(device)
    elif isinstance(model, str):
        if model == "toy_convnet":
            # toy model
            input_shape = tuple(next(iter(train_data))[0].shape[1:])
            model = ToyTorchConvnet(input_shape, num_classes).to(device)
        elif model == "toy_mlp":
            # toy model
            input_shape = tuple(next(iter(train_data))[0].shape[1:])
            model = ToyTorchMLP(input_shape, num_classes).to(device)
        else:
            # torchvision model
            model = getattr(torchvision.models, model)(
                num_classes=num_classes, pretrained=imagenet_pretrained
            ).to(device)

    # define optimizer and learning rate scheduler
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate)
    n_steps = len(train_data) * epochs
    if lr_scheduler == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    elif lr_scheduler == "steps":
        boundaries = list(np.round(n_steps * np.array([1 / 3, 2 / 3])).astype(int))
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
    model: nn.Module,
    train_data: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    save_dir: Optional[str] = None,
    validation_data: Optional[DataLoader] = None,
) -> nn.Module:
    """Torch basic training loop

    Args:
        model (nn.Module): Model to train.
        train_data (DataLoader): Train dataloader.
        epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.modules.loss._Loss): Criterion for loss.
        device (torch.device): On which device to train (CUDA or CPU).
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            Defaults to None.
        save_dir (str, optional): Where the model will be saved. Defaults to None.
        validation_data (DataLoader, optional): Validation dataloader. Defaults to None.

    Returns:
        nn.Module: Trained model.
    """
    best_val_acc = None
    for epoch in range(epochs):
        # train phase
        model.train()
        running_loss, running_acc = 0.0, 0.0
        with tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs} [Train]") as iterator:
            for i, (inputs, labels) in enumerate(iterator):
                # assign [inputs, labels] tensors to GPU
                inputs = inputs.to(device)
                labels = labels.long().to(device)

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

        if lr_scheduler is not None:
            lr_scheduler.step()

        # validation phase
        if validation_data is not None:
            model.eval()
            running_loss, running_acc = 0.0, 0.0
            with tqdm(
                validation_data, desc=f"Epoch {epoch + 1}/{epochs} [Val]"
            ) as iterator:
                for i, (inputs, labels) in enumerate(iterator):
                    # assign [inputs, labels] tensors to GPU
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

    if save_dir is not None:
        torch.save(model, os.path.join(save_dir, "last.pt"))
    return model
