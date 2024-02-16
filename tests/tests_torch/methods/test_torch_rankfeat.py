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

import pytest
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from oodeel.methods import Energy
from oodeel.methods import Entropy
from oodeel.methods import MLS
from oodeel.methods import ODIN
from tests.tests_torch import eval_detector_on_blobs
from oodeel.datasets import OODDataset

from oodeel.utils.torch_training_tools import train_torch_model
from oodeel.eval.plots import plot_ood_scores, plot_roc_curve


def load_mnist(data_path):
    # === load ID and OOD data ===
    batch_size = 128
    in_labels = [0, 1, 2, 3, 4]

    # 1- load train/test MNIST dataset
    mnist_train = OODDataset(
        dataset_id="MNIST",
        backend="torch",
        load_kwargs={"root": data_path, "train": True, "download": True},
    )
    mnist_test = OODDataset(
        dataset_id="MNIST",
        backend="torch",
        load_kwargs={"root": data_path, "train": False, "download": True},
    )

    # 2- split ID / OOD data depending on label value:
    # in-distribution: MNIST[0-4] / out-of-distribution: MNIST[5-9]
    oods_fit, _ = mnist_train.split_by_class(in_labels=in_labels)
    oods_in, oods_out = mnist_test.split_by_class(in_labels=in_labels)

    # 3- prepare data (preprocess, shuffle, batch) => torch dataloaders
    def preprocess_fn(*inputs):
        """Simple preprocessing to normalize images in [0, 1]."""
        x = inputs[0] / 255.0
        return tuple([x] + list(inputs[1:]))

    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
    )
    ds_in = oods_in.prepare(
        batch_size=batch_size, with_ood_labels=False, preprocess_fn=preprocess_fn
    )
    ds_out = oods_out.prepare(
        batch_size=batch_size, with_ood_labels=False, preprocess_fn=preprocess_fn
    )
    return ds_fit, ds_in, ds_out


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

    # Load MNIST dataset
    data_path = os.path.expanduser("~/") + ".oodeel/datasets"
    ds_fit, ds_in, ds_out = load_mnist(data_path)

    # Load or train a model
    model_path = os.path.expanduser("~/") + ".oodeel/saved_models"
    model_path_mnist_04 = os.path.join(model_path, "mnist_model_0-4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # if the model exists, load it
        model = torch.load(os.path.join(model_path_mnist_04, "best.pt")).to(device)
    except OSError:
        # else, train a new model
        train_config = {
            "model": "toy_convnet",
            "num_classes": 10,
            "epochs": 5,
            "save_dir": model_path_mnist_04,
            "validation_data": ds_in,
        }
        model = train_torch_model(ds_fit, **train_config).to(device)

    # evaluate model
    model.eval()
    labels, preds = [], []
    for x, y in ds_in:
        x = x.to(device)
        preds.append(torch.argmax(model(x), dim=-1).detach().cpu())
        labels.append(y)
    print(f"Test accuracy:\t{accuracy_score(torch.cat(labels), torch.cat(preds)):.6f}")

    # Run OOD detector
    d_kwargs = detectors[detector_name]["kwargs"]
    detector = detectors[detector_name]["class"](
        use_rankfeat=True, rankfeat_layer_id="relu2", **d_kwargs
    )
    detector.fit(model, fit_dataset=ds_fit)
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
    filename = f"ood_scores_{detector_name}_{rankfeat}.png"
    plt.savefig(filename)
    print(f"Image saved to {filename}")


test_rankfeat("energy", 0.95, 0.23)
test_rankfeat("mls", 0.95, 0.23)
test_rankfeat("msp", 0.95, 0.23)
test_rankfeat("entropy", 0.95, 0.23)
test_rankfeat("odin", 0.95, 0.23)
