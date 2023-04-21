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
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from oodeel.datasets import OODDataset
from oodeel.eval.metrics import bench_metrics
from oodeel.methods import DKNN
from oodeel.methods.mahalanobis import Mahalanobis

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

pp = pprint.PrettyPrinter()


@dataclass
class TFDSPytorchConverter(IterableDataset):
    tf_ds: tf.data.Dataset

    def __iter__(self) -> Iterator:
        for batch_x, batch_y in self.tf_ds.as_numpy_iterator():
            for sample_idx in range(batch_x.shape[0]):
                yield np.stack(
                    [batch_x[sample_idx, :, :, 0] for _ in range(3)], axis=0
                ), batch_y[sample_idx]


if __name__ == "__main__":
    oods_in = OODDataset("mnist", split="test")
    oods_out = OODDataset("fashion_mnist", split="test")
    oods_fit = OODDataset("mnist", split="train")

    def preprocess_fn(*inputs):
        x = inputs[0] / 255
        return tuple([x] + list(inputs[1:]))

    batch_size = 8
    ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_out = oods_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
    ds_fit = oods_fit.prepare(
        batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
    )

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
    )

    ds_fit_pt = DataLoader(TFDSPytorchConverter(ds_fit.take(1_000)), batch_size=8)
    ds_in_pt = DataLoader(TFDSPytorchConverter(ds_in.take(500)), batch_size=8)
    ds_out_pt = DataLoader(TFDSPytorchConverter(ds_out.take(500)), batch_size=8)

    ### DKNN

    oodmodel = DKNN(nearest=10, output_layers_id=["avgpool"])
    oodmodel.fit(model, ds_fit_pt)
    scores_in = oodmodel.score(ds_in_pt)
    scores_out = oodmodel.score(ds_out_pt)

    metrics = bench_metrics(
        (scores_in, scores_out),
        metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
        threshold=None,
    )

    pp.pprint(metrics)

    ## Mahalanobis

    oodmodel = Mahalanobis(
        output_layers_id=["avgpool"], input_processing_magnitude=0.0, mode="sklearn"
    )
    oodmodel.fit(model, ds_fit_pt)
    scores_in = oodmodel.score(ds_in_pt)
    scores_out = oodmodel.score(ds_out_pt)

    metrics = bench_metrics(
        (scores_in, scores_out),
        metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
        threshold=None,
    )

    pp.pprint(metrics)

    oodmodel = Mahalanobis(output_layers_id=["avgpool"], input_processing_magnitude=0.0)
    oodmodel.fit(model, ds_fit_pt)
    for mag in [0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:

        print(f"Magnitude : {mag}")

        oodmodel.input_processing_magnitude = mag
        scores_in = oodmodel.score(ds_in_pt)
        scores_out = oodmodel.score(ds_out_pt)

        metrics = bench_metrics(
            (scores_in, scores_out),
            metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
            threshold=None,
        )

        pp.pprint(metrics)
