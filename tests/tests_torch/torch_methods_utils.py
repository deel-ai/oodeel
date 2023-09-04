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

import numpy as np
import requests
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from oodeel.datasets import OODDataset
from oodeel.eval.metrics import bench_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.expanduser("~/") + ".oodeel/saved_models"
data_path = os.path.expanduser("~/") + ".oodeel/datasets"
os.makedirs(model_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)


def load_blobs_data(batch_size=128, num_samples=10000, train_ratio=0.8):
    # === data hparams ===
    num_classes = 3
    in_labels = [0, 1]
    out_labels = [2, 3]
    centers = np.array([[-4, -4], [4, 4], [-4, 4], [4, -4]])

    # === generate data ===
    X, y = make_blobs(num_samples, num_classes, centers=centers, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=0
    )

    # === id / ood split ===
    blobs_train = OODDataset((X_train, y_train), backend="torch")
    blobs_test = OODDataset((X_test, y_test), backend="torch")
    oods_fit, _ = blobs_train.split_by_class(in_labels, out_labels)
    oods_in, oods_out = blobs_test.split_by_class(in_labels, out_labels)

    # === prepare data (shuffle, batch) => torch dataloaders ===
    ds_fit = oods_fit.prepare(batch_size=batch_size, shuffle=True)
    ds_in = oods_in.prepare(batch_size=batch_size)
    ds_out = oods_out.prepare(batch_size=batch_size)
    return ds_fit, ds_in, ds_out


def load_blob_mlp():
    model_path_blob = os.path.join(model_path, "blobs_mlp.pt")

    # if model not in local, download it
    if not os.path.exists(model_path_blob):
        data = requests.get(
            "https://share.deel.ai/s/xcyk3ET8fzfTp8S/download/blobs_mlp.pt"
        )
        with open(model_path_blob, "wb") as file:
            file.write(data.content)

    # load model
    model = torch.load(model_path_blob, map_location=device)
    model.eval()
    return model


def eval_detector_on_blobs(
    detector,
    auroc_thr=0.6,
    fpr95_thr=0.3,
    batch_size=128,
    check_react_clipping=False,
):
    # seed
    torch.manual_seed(1)

    # load data
    ds_fit, ds_in, ds_out = load_blobs_data(batch_size)

    # get classifier
    model = load_blob_mlp()

    # fit ood detector
    if detector.requires_to_fit_dataset or detector.use_react:
        detector.fit(model, feature_layers_id=[-2], fit_dataset=ds_fit)
    else:
        detector.fit(model)

    # ood scores
    scores_in, info_in = detector.score(ds_in)
    scores_out, info_out = detector.score(ds_out)
    assert scores_in.shape == (1028,)
    assert info_in["labels"].shape == (1028,)
    assert info_in["logits"].shape == (1028, 2)
    assert scores_out.shape == (972,)
    assert info_out["labels"].shape == (972,)
    assert info_out["logits"].shape == (972, 2)

    # ood metrics: auroc, fpr95tpr
    metrics = bench_metrics(
        (scores_in, scores_out),
        metrics=["auroc", "fpr95tpr"],
    )
    auroc, fpr95tpr = metrics["auroc"], metrics["fpr95tpr"]
    assert auroc >= auroc_thr, f"got a score of {auroc}, below {auroc_thr}!"
    assert fpr95tpr <= fpr95_thr, f"got a score of {fpr95tpr}, above {fpr95_thr}!"

    # react specific test
    # /!\ do it at the end of the test because it may affect the detector's behaviour
    if check_react_clipping:
        assert detector.react_threshold is not None
        penult_feat_extractor = detector._load_feature_extractor(
            model=model, feature_layers_id=[-2, -1]
        )
        penult_features = penult_feat_extractor.predict(ds_fit)[0][0]
        assert torch.max(penult_features) <= detector.react_threshold, (
            f"Maximum value of penultimate features ({torch.max(penult_features)})"
            + " should be less than or equal to the react threshold value"
            + f" ({detector.react_threshold})"
        )
