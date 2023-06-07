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
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from oodeel.extractor.torch_feature_extractor import TorchFeatureExtractor
from tests.tests_torch import ComplexNet
from tests.tests_torch import generate_data_torch
from tests.tests_torch import named_sequential_model
from tests.tests_torch import Net
from tests.tests_torch import sequential_model

# From Pytorch CIFAR-10 example


@pytest.mark.parametrize(
    "kwargs_factory,expected_sz",
    [
        (lambda: dict(model=Net(), output_layers_id=["fc2"]), [100, 84]),
        (lambda: dict(model=sequential_model(), output_layers_id=[-2]), [100, 84]),
        (
            lambda: dict(model=named_sequential_model(), output_layers_id=["fc2"]),
            [100, 84],
        ),
        (lambda: dict(model=ComplexNet(), output_layers_id=["fcs.fc2"]), [100, 84]),
        (
            lambda: dict(model=ComplexNet(), output_layers_id=["fcs.fc2"]),
            [100, 84],
        ),
    ],
    ids=[
        "Pytorch simple Net",
        "Sequential model",
        "Sequential model with names",
        "Complex Pytorch model with layered layers",
        "Complex model with batch size",
    ],
)
def test_params_torch_feature_extractor(kwargs_factory, expected_sz):
    n_samples = 100
    input_shape = (3, 32, 32)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples // 2)

    feature_extractor = TorchFeatureExtractor(**kwargs_factory())
    pred_feature_extractor = feature_extractor.predict(dataset)

    assert list(pred_feature_extractor.size()) == expected_sz


@pytest.mark.parametrize(
    "kwargs_factory,expected_sz",
    [
        (
            lambda: dict(
                model=sequential_model(), input_layer_id=4, output_layers_id=[-2]
            ),
            [100, 84],
        ),
        (
            lambda: dict(
                model=named_sequential_model(),
                input_layer_id="conv2",
                output_layers_id=["fc2"],
            ),
            [100, 84],
        ),
    ],
    ids=["Sequential model", "Sequential model with names"],
)
def test_pytorch_feature_extractor_with_input_ids(kwargs_factory, expected_sz):
    n_samples = 100
    n_samples = 100
    input_shape = (6, 14, 14)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples // 2)

    feature_extractor = TorchFeatureExtractor(**kwargs_factory())
    pred_feature_extractor = feature_extractor.predict(dataset)

    assert list(pred_feature_extractor.size()) == expected_sz


def test_get_weights():
    model = named_sequential_model()

    model_fe = TorchFeatureExtractor(model, output_layers_id=[-1])
    W, b = model_fe.get_weights(-1)

    assert W.shape == (10, 84)
    assert b.shape == (10,)


def test_postproc_fns():
    n_samples = 100
    input_shape = (3, 32, 32)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples // 2)

    model = named_sequential_model()

    def globalavg(x):
        _, _, height, width = x.size()
        return nn.AvgPool2d(height, width)(x)

    postproc_fns = [globalavg, None]
    feature_extractor = TorchFeatureExtractor(
        model, output_layers_id=["relu2", "fc2"], postproc_fns=postproc_fns
    )

    feat0, feat1 = feature_extractor.predict(dataset)
    assert list(feat0.size()) == [100, 16, 1, 1]
    assert list(feat1.size()) == [100, 84]
