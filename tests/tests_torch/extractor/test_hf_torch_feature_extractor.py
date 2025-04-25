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
from transformers import MobileNetV1ForImageClassification

from oodeel.extractor.hf_torch_feature_extractor import HFTorchFeatureExtractor
from tests.tests_torch import generate_data_torch

# From Pytorch CIFAR-10 example


@pytest.mark.parametrize(
    "kwargs_factory,expected_sz",
    [
        (lambda: dict(feature_layers_id=["hidden_0"]), [[100, 32, 112, 112]]),
        (lambda: dict(feature_layers_id=[-2]), [[100, 1024]]),
        (
            lambda: dict(feature_layers_id=["hidden_0", -2]),
            [[100, 32, 112, 112], [100, 1024]],
        ),
    ],
    ids=[
        "with only HF hidden states",
        "without HF hidden states",
        "with both HF hidden states and user specified layers",
    ],
)
def test_params_torch_feature_extractor(kwargs_factory, expected_sz):
    n_samples = 100
    input_shape = (3, 224, 224)
    num_labels = 10

    model = MobileNetV1ForImageClassification.from_pretrained(
        "google/mobilenet_v1_1.0_224"
    )
    x = generate_data_torch(input_shape, num_labels, n_samples)

    dataset = DataLoader(x, batch_size=n_samples // 2)

    feature_extractor = HFTorchFeatureExtractor(model, **kwargs_factory())
    pred_feature_extractor, _ = feature_extractor.predict(dataset)

    if len(expected_sz) > 1:
        assert list(pred_feature_extractor[1].size()) == expected_sz[1]
    assert list(pred_feature_extractor[0].size()) == expected_sz[0]


def test_postproc_fns():
    n_samples = 100
    input_shape = (3, 224, 224)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples // 2)

    model = MobileNetV1ForImageClassification.from_pretrained(
        "google/mobilenet_v1_1.0_224"
    )

    def globalavg(x):
        _, _, height, width = x.size()
        return nn.AvgPool2d(height, width)(x)

    postproc_fns = [globalavg, globalavg, globalavg, lambda x: x]
    feature_extractor = HFTorchFeatureExtractor(
        model,
        feature_layers_id=[
            "hidden_0",
            "mobilenet_v1.layer.25.activation",
            "hidden_1",
            -2,
        ],
    )

    feats, _ = feature_extractor.predict(dataset, postproc_fns=postproc_fns)
    feat0, feat1, feat2, feat3 = feats
    assert list(feat0.size()) == [100, 32, 1, 1]
    assert list(feat1.size()) == [100, 1024, 1, 1]
    assert list(feat2.size()) == [100, 64, 1, 1]
    assert list(feat3.size()) == [100, 1024]
