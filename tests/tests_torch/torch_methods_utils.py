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

import torch

from oodeel.datasets import OODDataset
from oodeel.eval.metrics import bench_metrics
from oodeel.utils.torch_training_tools import train_torch_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.expanduser("~/") + ".oodeel/saved_models"
data_path = os.path.expanduser("~/") + ".oodeel/datasets"
os.makedirs(model_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)


def load_mnist_vs_fmnist(batch_size=128):
    # load mnist train / test
    oods_fit = OODDataset(
        dataset_id="MNIST",
        backend="torch",
        load_kwargs={"root": data_path, "train": True, "download": True},
    )
    oods_in = OODDataset(
        dataset_id="MNIST",
        backend="torch",
        load_kwargs={"root": data_path, "train": False, "download": True},
    )
    oods_out = OODDataset(
        dataset_id="FashionMNIST",
        backend="torch",
        load_kwargs={"root": data_path, "train": False, "download": True},
    )

    # prepare data (preprocess, shuffle, batch) => torch dataloaders
    def preprocess_fn(inputs):
        """Simple preprocessing function to normalize images in [0, 1]."""
        x = inputs[0] / 255.0
        return tuple([x] + list(inputs[1:]))

    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
    )
    ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_out = oods_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    return ds_fit, ds_in, ds_out


def get_mnist_toy_convnet(ds_fit, ds_in):
    model_path_mnist = os.path.join(model_path, "mnist_model")

    if os.path.exists(os.path.join(model_path_mnist, "best.pt")):
        # if the model exists, load it
        model = torch.load(
            os.path.join(model_path_mnist, "best.pt"), map_location=device
        )
    else:
        # else, train a new model
        train_config = {
            "model": "toy_convnet",
            "num_classes": 10,
            "epochs": 5,
            "save_dir": model_path_mnist,
            "validation_data": ds_in,
        }
        model = train_torch_model(ds_fit, **train_config)
    model.eval()
    return model


def eval_detector_on_mnist(
    detector, need_to_fit_dataset, auroc_thr=0.6, fpr95_thr=0.3, batch_size=128
):
    # seed
    torch.manual_seed(0)

    # load data
    ds_fit, ds_in, ds_out = load_mnist_vs_fmnist(batch_size)

    # get classifier
    model = get_mnist_toy_convnet(ds_fit, ds_in)

    # fit ood detector
    if need_to_fit_dataset:
        detector.fit(model, ds_fit)
    else:
        detector.fit(model)

    # ood scores
    scores_in = detector.score(ds_in)
    scores_out = detector.score(ds_out)
    assert scores_in.shape == (10000,)
    assert scores_out.shape == (10000,)

    # ood metrics: auroc, fpr95tpr
    metrics = bench_metrics(
        (scores_in, scores_out),
        metrics=["auroc", "fpr95tpr"],
    )
    assert metrics["auroc"] >= auroc_thr
    assert metrics["fpr95tpr"] <= fpr95_thr