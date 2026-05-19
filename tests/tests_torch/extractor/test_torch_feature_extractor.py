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
import torch
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
        (lambda: dict(model=Net(), feature_layers_id=["fc2"]), [100, 84]),
        (lambda: dict(model=sequential_model(), feature_layers_id=[8]), [100, 84]),
        (
            lambda: dict(model=named_sequential_model(), feature_layers_id=["fc2"]),
            [100, 84],
        ),
        (lambda: dict(model=ComplexNet(), feature_layers_id=["fcs.fc2"]), [100, 84]),
        (
            lambda: dict(model=ComplexNet(), feature_layers_id=["fcs.fc2"]),
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
    pred_feature_extractor, _ = feature_extractor.predict(dataset)

    assert list(pred_feature_extractor[0].size()) == expected_sz


@pytest.mark.parametrize(
    "kwargs_factory,expected_sz",
    [
        (
            lambda: dict(
                model=sequential_model(), input_layer_id=4, feature_layers_id=[-2]
            ),
            [100, 84],
        ),
        (
            lambda: dict(
                model=named_sequential_model(),
                input_layer_id="conv2",
                feature_layers_id=["fc2"],
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
    pred_feature_extractor, _ = feature_extractor.predict(dataset)

    assert list(pred_feature_extractor[0].size()) == expected_sz


def test_get_weights():
    model = named_sequential_model()

    model_fe = TorchFeatureExtractor(model, feature_layers_id=[-1])
    W, b = model_fe.get_weights(-1)

    assert W.shape == (10, 84)
    assert b.shape == (10,)


def test_predict_with_labels():
    """Assert that FeatureExtractor.predict() correctly returns features and labels when
    return_labels=True.

    Multiple tests are performed:
    - dataset with labels or without labels
    - dataset with one-hot encoded or sparse labels
    - single tensor instead of a dataset
    """
    input_shape = (3, 32, 32)
    num_labels = 10
    n_samples = 100

    # Generate dataset with sparse labels, with one-hot labels and without labels
    x = generate_data_torch(input_shape, num_labels, n_samples, one_hot=False)
    dataset = DataLoader(x, batch_size=n_samples // 3)

    x_one_hot = generate_data_torch(input_shape, num_labels, n_samples)
    dataset_one_hot = DataLoader(x_one_hot, batch_size=n_samples // 3)

    x_wo_labels = generate_data_torch(
        input_shape, num_labels, n_samples, with_labels=False
    )
    dataset_wo_labels = DataLoader(x_wo_labels, batch_size=n_samples // 3)

    # Generate model and feature extractor
    model = ComplexNet()
    feature_extractor = TorchFeatureExtractor(model, feature_layers_id=["fcs.fc2"])

    # Assert predict() outputs have expected shape
    out, info = feature_extractor.predict(dataset)
    assert out[0].shape == (n_samples, 84)
    assert info["logits"].shape == (n_samples, 10)
    assert info["labels"].shape == (n_samples,)

    # Assert predict() outputs have expected shape (dataset has one-hot encoded labels)
    out, info = feature_extractor.predict(dataset_one_hot)
    assert out[0].shape == (n_samples, 84)
    assert info["logits"].shape == (n_samples, 10)
    assert info["labels"].shape == (n_samples,)

    # Assert predict() outputs have expected shape (dataset has no labels)
    out, info = feature_extractor.predict(dataset_wo_labels)
    assert out[0].shape == (n_samples, 84)
    assert info["logits"].shape == (n_samples, 10)
    assert info["labels"] is None

    # Assert predict() outputs for a single input tensor (no label provided)
    batch = next(iter(dataset_wo_labels))
    out, info = feature_extractor.predict(batch)
    assert out[0].shape == (33, 84)
    assert info["logits"].shape == (33, 10)
    assert info["labels"] is None

    # Assert predict() outputs for a single input tensor with label provided
    batch = next(iter(dataset_one_hot))
    out, info = feature_extractor.predict(batch)
    assert out[0].shape == (33, 84)
    assert info["logits"].shape == (33, 10)
    assert info["labels"].shape == (33,)


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

    postproc_fns = [globalavg, lambda x: x]
    feature_extractor = TorchFeatureExtractor(model, feature_layers_id=["relu2", "fc2"])

    feats, _ = feature_extractor.predict(dataset, postproc_fns=postproc_fns)
    feat0, feat1 = feats
    assert list(feat0.size()) == [100, 16, 1, 1]
    assert list(feat1.size()) == [100, 84]


def test_multiple_extractors_hook_isolation():
    """Test that multiple extractors on the same model don't interfere with each other.

    This test verifies that when two TorchFeatureExtractor instances are created
    on the same model, each extractor only captures features from its own hooks
    when predict is called, without being affected by the other extractor's hooks.
    """
    n_samples = 20
    input_shape = (3, 32, 32)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples)

    model = Net()

    # Create first extractor targeting fc1
    extractor1 = TorchFeatureExtractor(model, feature_layers_id=["fc1"])

    # Create second extractor targeting fc2
    extractor2 = TorchFeatureExtractor(model, feature_layers_id=["fc2"])

    # Get features from extractor1
    features1, _ = extractor1.predict(dataset)

    # Get features from extractor2
    features2, _ = extractor2.predict(dataset)

    # Verify each extractor captured the correct layer dimensions
    # fc1 outputs 120 features, fc2 outputs 84 features
    assert features1[0].shape == (n_samples, 120), "Extractor1 should capture fc1 (120)"
    assert features2[0].shape == (n_samples, 84), "Extractor2 should capture fc2 (84)"

    # Verify extractor1 still works after extractor2 was used
    features1_again, _ = extractor1.predict(dataset)
    assert torch.allclose(
        features1[0], features1_again[0]
    ), "Extractor1 should produce same results after extractor2 was used"


def test_extractor_clean_hooks():
    """Test that clean_hooks properly removes hooks from the model."""
    n_samples = 10
    input_shape = (3, 32, 32)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples)

    model = Net()

    # Create extractor and verify it works
    extractor = TorchFeatureExtractor(model, feature_layers_id=["fc1"])
    features, _ = extractor.predict(dataset)
    assert features[0].shape == (n_samples, 120)

    # Count hooks before cleanup
    num_handles_before = len(extractor._handles)
    assert num_handles_before > 0, "Extractor should have registered hooks"

    # Clean hooks
    extractor.clean_hooks()

    # Verify that model's hooks are removed
    for module in model.modules():
        assert not module._forward_hooks, "All forward hooks should be removed"
        assert not module._backward_hooks, "All backward hooks should be removed"
        assert not module._forward_pre_hooks, "All forward pre-hooks should be removed"

    # Verify hooks are removed
    assert len(extractor._handles) == 0, "Hooks should be removed after clean_hooks"


def test_react_hook_isolation():
    """Test that ReAct hooks are also isolated between extractors."""
    n_samples = 10
    input_shape = (3, 32, 32)
    num_labels = 10

    x = generate_data_torch(input_shape, num_labels, n_samples)
    dataset = DataLoader(x, batch_size=n_samples)

    model = Net()

    # Create extractor with ReAct (clips activations)
    extractor_react = TorchFeatureExtractor(
        model, feature_layers_id=["fc2"], react_threshold=1.0
    )

    # Create normal extractor
    extractor_normal = TorchFeatureExtractor(model, feature_layers_id=["fc2"])

    # Get features from normal extractor
    features_normal, _ = extractor_normal.predict(dataset)

    # Get features from react extractor
    features_react, _ = extractor_react.predict(dataset)

    # Get features from normal extractor again - should be unaffected by react
    features_normal_again, _ = extractor_normal.predict(dataset)

    # Normal extractor should produce same results before and after react was used
    assert torch.allclose(
        features_normal[0], features_normal_again[0]
    ), "Normal extractor should not be affected by ReAct extractor"
