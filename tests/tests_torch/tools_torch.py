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
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def almost_equal(arr1, arr2, epsilon=1e-6):
    """Ensure two array are almost equal at an epsilon"""
    return np.mean(np.abs(arr1 - arr2)) < epsilon


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ComplexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 6, 5)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("conv2", nn.Conv2d(6, 16, 5)),
                    ("relu2", nn.ReLU()),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        self.fcs = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(16 * 5 * 5, 120)),
                    ("fc2", nn.Linear(120, 84)),
                    ("fc3", nn.Linear(84, 10)),
                ]
            )
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fcs(x)
        return x


def sequential_model():
    return nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Linear(120, 84),
        nn.Linear(84, 10),
    )


def named_sequential_model():
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(3, 6, 5)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(2, 2)),
                ("conv2", nn.Conv2d(6, 16, 5)),
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(2, 2)),
                ("flatten", nn.Flatten()),
                ("fc1", nn.Linear(16 * 5 * 5, 120)),
                ("fc2", nn.Linear(120, 84)),
                ("fc3", nn.Linear(84, 10)),
            ]
        )
    )


def simplest_mlp(num_features, num_classes):
    return nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


def generate_data(x_shape=(3, 32, 32), num_labels=10, samples=100, one_hot=True):
    x = np.random.rand(samples, *x_shape).astype(np.float32)
    x /= np.max(x)
    if one_hot:
        y = np.eye(num_labels)[np.random.randint(0, num_labels, samples)]
    else:
        y = np.random.randint(0, num_labels, samples)

    return x, y


def generate_data_torch(
    x_shape=(3, 32, 32), num_labels=10, samples=100, one_hot=True, with_labels=True
):
    x, y = generate_data(x_shape, num_labels, samples, one_hot)
    if with_labels:
        return TensorDataset(torch.Tensor(x), torch.Tensor(y))
    else:
        return TensorDataset(torch.Tensor(x))
