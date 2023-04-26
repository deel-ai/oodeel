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
import pprint
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import torch
from torchvision import transforms

from oodeel.datasets import OODDataset
from oodeel.eval.metrics import bench_metrics
from oodeel.methods.mahalanobis import Mahalanobis

warnings.filterwarnings("ignore")


pp = pprint.PrettyPrinter()

model_path = os.path.expanduser("~/") + ".oodeel/saved_models"
data_path = os.path.expanduser("~/") + ".oodeel/datasets"
os.makedirs(model_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)


def torch_main(batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Mahalanobis demo (torch) ===")
    # === EXP 1 ===
    print("--- CIFAR10 vs SVHN ---")
    oods_in = OODDataset(
        "CIFAR10", backend="torch", load_kwargs={"root": data_path, "train": False}
    )
    oods_out = OODDataset(
        "SVHN", backend="torch", load_kwargs={"root": data_path, "split": "test"}
    )
    oods_fit = OODDataset(
        "CIFAR10", backend="torch", load_kwargs={"root": data_path, "train": True}
    )

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
    ).to(device)

    def preprocess_fn(inputs):
        x = inputs[0] / 255
        x = transforms.Normalize(
            mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]
        )(x)
        return tuple([x] + list(inputs[1:]))

    ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_out = oods_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
    )
    # compute mahalanobis scores
    eps = 0.002
    print(f"Magnitude : {eps}")
    oodmodel = Mahalanobis(output_layers_id=["avgpool"], eps=eps)
    oodmodel.fit(model, ds_fit)
    scores_in = oodmodel.score(ds_in)
    scores_out = oodmodel.score(ds_out)

    metrics = bench_metrics(
        (scores_in, scores_out), metrics=["auroc", "tnr95tpr", "detect_acc"]
    )
    pp.pprint(metrics)

    # === EXP 2 ===
    print("--- MNIST vs FashionMNIST ---")
    oods_in = OODDataset(
        "MNIST", backend="torch", load_kwargs={"root": data_path, "train": False}
    )
    oods_out = OODDataset(
        "FashionMNIST", backend="torch", load_kwargs={"root": data_path, "train": False}
    )
    oods_fit = OODDataset(
        "MNIST", backend="torch", load_kwargs={"root": data_path, "train": True}
    )

    model = torch.load(
        os.path.join(model_path, "mnist_model/best.pt"), map_location=device
    )

    def preprocess_fn(inputs):
        x = inputs[0] / 255
        return tuple([x] + list(inputs[1:]))

    ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_out = oods_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
    )
    # compute mahalanobis scores
    eps = 0.002
    print(f"Magnitude : {eps}")
    oodmodel = Mahalanobis(output_layers_id=[-2], eps=eps)
    oodmodel.fit(model, ds_fit)
    scores_in = oodmodel.score(ds_in)
    scores_out = oodmodel.score(ds_out)

    metrics = bench_metrics(
        (scores_in, scores_out), metrics=["auroc", "tnr95tpr", "detect_acc"]
    )
    pp.pprint(metrics)


def tf_main(batch_size=8):
    print("=== Mahalanobis demo (tensorflow) ===")
    # === EXP 1 ===
    print("--- MNIST vs FashionMNIST ---")
    oods_in = OODDataset(
        "mnist", backend="tensorflow", input_key="image", load_kwargs={"split": "test"}
    )
    oods_out = OODDataset(
        "fashion_mnist",
        backend="tensorflow",
        input_key="image",
        load_kwargs={"split": "test"},
    )
    oods_fit = OODDataset(
        "mnist", backend="tensorflow", input_key="image", load_kwargs={"split": "train"}
    )

    model = tf.keras.models.load_model(os.path.join(model_path, "mnist_model.h5"))

    def preprocess_fn(*inputs):
        x = inputs[0] / 255
        return tuple([x] + list(inputs[1:]))

    ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_out = oods_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
    )
    # compute mahalanobis scores
    eps = 0.002
    print(f"Magnitude : {eps}")
    oodmodel = Mahalanobis(output_layers_id=[-2], eps=eps)
    oodmodel.fit(model, ds_fit.take(100))
    scores_in = oodmodel.score(ds_in.take(100))
    scores_out = oodmodel.score(ds_out.take(100))

    metrics = bench_metrics(
        (scores_in, scores_out), metrics=["auroc", "tnr95tpr", "detect_acc"]
    )
    pp.pprint(metrics)


if __name__ == "__main__":
    torch_main()
    tf_main()
