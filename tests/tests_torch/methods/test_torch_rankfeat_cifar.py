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

import matplotlib.pyplot as plt
import pytest
import torch
from sklearn.metrics import accuracy_score
from torchvision import transforms

from oodeel.datasets import OODDataset
from oodeel.eval.plots import plot_ood_scores, plot_roc_curve
from oodeel.methods import MLS, ODIN, Energy, Entropy
from oodeel.utils.torch_training_tools import train_torch_model
from tests.tests_torch import eval_detector_on_blobs


def load_cifar(data_path):
    # === load ID and OOD data ===
    batch_size = 128

    # 1a- load in-distribution dataset: CIFAR-10
    oods_in = OODDataset(
        dataset_id="CIFAR10",
        backend="torch",
        load_kwargs={"root": data_path, "train": False, "download": True},
    )
    # 1b- load out-of-distribution dataset: SVHN
    oods_out = OODDataset(
        dataset_id="SVHN",
        backend="torch",
        load_kwargs={"root": data_path, "split": "test", "download": True},
    )

    # 2- prepare data (preprocess, shuffle, batch) => torch dataloaders
    def preprocess_fn(*inputs):
        """Preprocessing function from
        https://github.com/chenyaofo/pytorch-cifar-models
        """
        x = inputs[0] / 255.0
        x = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x)
        return tuple([x] + list(inputs[1:]))

    ds_in = oods_in.prepare(batch_size, preprocess_fn)
    ds_out = oods_out.prepare(batch_size, preprocess_fn)
    return ds_in, ds_out


@pytest.mark.parametrize(
    "detector_name,auroc_thr,fpr95_thr",
    [
        # ("odin", 0.95, 0.05),
        # ("mls", 0.95, 0.05),
        # ("msp", 0.95, 0.05),
        ("energy", 0.95, 0.23),
        # ("entropy", 0.95, 0.05),
    ],
)
def test_rankfeat(detector_name, auroc_thr, fpr95_thr):
    """
    Test RankFeat + [MLS, MSP, Energy, ODIN, Entropy] on toy blobs OOD dataset-wise task

    We check that the area under ROC is above a certain threshold, and that the FPR95TPR
    is below an other threshold.
    """
    detectors = {
        "odin": {
            "class": ODIN,
            "kwargs": dict(temperature=1000),
        },
        "mls": {
            "class": MLS,
            "kwargs": dict(),
        },
        "msp": {
            "class": MLS,
            "kwargs": dict(output_activation="softmax"),
        },
        "energy": {
            "class": Energy,
            "kwargs": dict(),
        },
        "entropy": {
            "class": Entropy,
            "kwargs": dict(),
        },
    }

    # Load CIFAR/SVHN dataset
    data_path = os.path.expanduser("~/") + ".oodeel/datasets"
    ds_in, ds_out = load_cifar(data_path)

    # Load model: resnet20 pretrained on CIFAR-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model="cifar10_resnet20",
        pretrained=True,
        verbose=False,
    ).to(device)
    model.eval()

    # evaluate model
    labels, preds = [], []
    for x, y in ds_in:
        x = x.to(device)
        preds.append(torch.argmax(model(x), dim=-1).detach().cpu())
        labels.append(y)
    print(f"Test accuracy:\t{accuracy_score(torch.cat(labels), torch.cat(preds)):.6f}")

    # Run OOD detector
    d_kwargs = detectors[detector_name]["kwargs"]
    detector = detectors[detector_name]["class"](
        use_rankfeat=True, rankfeat_layer_id="layer3", **d_kwargs
    )
    detector.fit(model)
    scores_in, _ = detector.score(ds_in)
    scores_out, _ = detector.score(ds_out)

    # Debug logging
    print(scores_in.shape, scores_out.shape)
    print(scores_in.mean(), scores_out.mean())

    log_scale = detector_name in ["msp", "entropy"]
    plt.figure(figsize=(9, 3))
    plt.subplot(121)
    plot_ood_scores(scores_in, scores_out, log_scale=log_scale)
    plt.subplot(122)
    plot_roc_curve(scores_in, scores_out)
    plt.tight_layout()

    rankfeat = "rankfeat" if detector.use_rankfeat else "no_rankfeat"
    filename = f"ood_scores_cifar_{detector_name}_{rankfeat}.png"
    plt.savefig(filename)
    print(f"Image saved to {filename}")


test_rankfeat("energy", 0.95, 0.23)
test_rankfeat("mls", 0.95, 0.23)
test_rankfeat("msp", 0.95, 0.23)
test_rankfeat("entropy", 0.95, 0.23)
test_rankfeat("odin", 0.95, 0.23)
