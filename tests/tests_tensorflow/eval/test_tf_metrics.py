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
import numpy as np
import pytest
from pytest import approx
from sklearn.metrics import accuracy_score, roc_auc_score

from oodeel.eval.metrics import bench_metrics
from oodeel.eval.metrics import ftpn
from oodeel.eval.metrics import get_curve


@pytest.mark.parametrize(
    "threshold,expected_results", [(0.5, [0, 3, 2, 5]), (0.1, [4, 5, 0, 1])]
)
def test_fptn(threshold, expected_results):
    labels = np.array([0.0] * 5 + [1.0] * 5)  # 5 ID + 5 OOD samples
    scores = np.array([0.0] + [0.2] * 6 + [1.0] * 3)  # ID: 1x0, 4x0.2 / OOD: 2x0.1, 3x1
    # get ftpn
    fp, tp, fn, tn = ftpn(scores, labels, threshold=threshold)
    # compare to expected values
    fp_gt, tp_gt, fn_gt, tn_gt = expected_results
    assert fp == approx(fp_gt)
    assert tp == approx(tp_gt)
    assert fn == approx(fn_gt)
    assert tn == approx(tn_gt)


def test_get_curve():
    labels = np.array([0.0] * 5 + [1.0] * 5)  # 5 ID + 5 OOD samples
    scores = np.array([0.0] + [0.2] * 6 + [1.0] * 3)  # ID: 1x0, 4x0.2 / OOD: 2x0.1, 3x1
    # get curves
    (fpc, tpc, fnc, tnc), (_, _, _, _, _) = get_curve(
        scores, labels, step=1, return_raw=True
    )
    # compare to expected values (hand calculated)
    fpc_gt = np.array([4] * 6 + [0] * 3)
    tpc_gt = np.array([5] * 6 + [3] * 3)
    fnc_gt = np.array([0] * 6 + [2] * 3)
    tnc_gt = np.array([1] * 6 + [5] * 3)
    assert np.all(fpc == fpc_gt)
    assert np.all(tpc == tpc_gt)
    assert np.all(fnc == fnc_gt)
    assert np.all(tnc == tnc_gt)


@pytest.mark.parametrize("in_value,out_value", [(0.0, 1.0), (2.0, 4.0)])
def test_bench_metrics(in_value, out_value):
    labels = np.array([in_value] * 5 + [out_value] * 5)  # 5 ID + 5 OOD samples
    scores = np.array([0.0] + [0.2] * 6 + [1.0] * 3)  # ID: 1x0, 4x0.2 / OOD: 2x0.1, 3x1
    metrics = [
        "auroc",
        "tpr50fpr",
        "fpr15tnr",
        "tnr90fpr",
        accuracy_score,
        roc_auc_score,
        "detect_acc",
    ]
    # get metrics
    metrics_dict = bench_metrics(
        scores,
        labels,
        in_value,
        out_value,
        metrics,
        threshold=0.5,
        step=1,
    )
    # expected results (hand calculated)
    assert metrics_dict["auroc"] == approx(1 - (0.4 * 0.8) / 2)
    assert metrics_dict["roc_auc_score"] == approx(1 - (0.4 * 0.8) / 2)
    assert metrics_dict["accuracy_score"] == approx(8 / 10)
    assert metrics_dict["detect_acc"] == approx(8 / 10)
    assert metrics_dict["tpr50fpr"] == 0.6
    assert metrics_dict["fpr15tnr"] == 0.8
    assert metrics_dict["tnr90fpr"] == 0.2

    # Assert scores as (scores_in, scores_out) return the same results
    scores_in = scores[labels == in_value]
    scores_out = scores[labels == out_value]
    metrics_dict_2 = bench_metrics(
        (scores_in, scores_out),
        metrics=metrics,
        threshold=0.5,
        step=1,
    )
    assert metrics_dict == metrics_dict_2
