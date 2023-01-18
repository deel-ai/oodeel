from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from oodeel.models.torch_feature_extractor import TorchFeatureExtractor


# From Pytorch CIFAR-10 example
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
                ])

        )

        self.fcs = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(16 * 5 * 5, 120)),
                    ("fc2", nn.Linear(120, 84)),
                    ("fc3", nn.Linear(84, 10))])

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
        nn.Linear(84, 10)
    )


def named_sequential_model():
    return nn.Sequential(OrderedDict(
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
            ("fc3", nn.Linear(84, 10))])
    )


"""
There are various ways to define models in Pytorch. The feature extractor is tested on usual ways of defining models. 
"""


@pytest.mark.parametrize("kwargs_factory,expected_sz", [
    (lambda: dict(model=Net(), output_layers_id=["fc2"], flatten=True), [100, 84]),
    (lambda: dict(model=Net(), output_layers_id=["fc2"], flatten=True, output_activation="ReLU"), [100, 84]),
    (lambda: dict(model=sequential_model(), flatten=True, output_layers_id=[-2]), [100, 84]),
    (lambda: dict(model=named_sequential_model(), flatten=True, output_layers_id=["fc2"]), [100, 84]),
    (lambda: dict(model=ComplexNet(), flatten=True, output_layers_id=["fcs.fc2"]), [100, 84]),
    (lambda: dict(model=ComplexNet(), flatten=True, output_layers_id=["fcs.fc2"], batch_size=8), [100, 84])

], ids=[
    "Pytorch simple Net",
    "Pytorch simple Net with activation",
    "Sequential model",
    "Sequential model with names",
    "Complex Pytorch model with layered layers",
    "Complex model with batch size"
])
def test_params_torch_feature_extractor(kwargs_factory, expected_sz):
    n_samples = 100
    n_channels = 3
    imsize = 32

    x = np.random.randn(n_samples, n_channels, imsize, imsize)
    x_tens = torch.from_numpy(x).float()

    feature_extractor = TorchFeatureExtractor(**kwargs_factory())
    pred_feature_extractor = feature_extractor.predict(x_tens)

    assert len(pred_feature_extractor) == 1
    assert list(pred_feature_extractor[0].size()) == expected_sz


@pytest.mark.parametrize("kwargs_factory,expected_sz", [
    (lambda: dict(model=sequential_model(), input_layer_id=4, output_layers_id=[-2]), [100, 84]),
    (lambda: dict(model=named_sequential_model(), input_layer_id="conv2", output_layers_id=["fc2"]), [100, 84]),
], ids=[
    "Sequential model",
    "Sequential model with names"
])
def test_pytorch_feature_extractor_with_input_ids(kwargs_factory, expected_sz):
    n_samples = 100
    imsize_at_layer = 14
    n_channels_at_layer = 6

    x = np.random.randn(n_samples, n_channels_at_layer, imsize_at_layer, imsize_at_layer)
    x_tens = torch.from_numpy(x).float()

    feature_extractor = TorchFeatureExtractor(**kwargs_factory())

    pred_feature_extractor = feature_extractor.predict(x_tens)

    assert len(pred_feature_extractor) == 1
    assert list(pred_feature_extractor[0].size()) == expected_sz
