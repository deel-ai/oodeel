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
import re

import numpy as np
import sklearn

from ..types import Optional
from ..types import Tuple
from ..types import Union


def bench_metrics(
    scores: Union[np.ndarray, tuple],
    labels: Optional[np.ndarray] = None,
    in_value: Optional[int] = 0,
    out_value: Optional[int] = 1,
    metrics: Optional[list] = ["auroc", "fpr95tpr"],
    threshold: Optional[float] = None,
    step: Optional[int] = 4,
) -> dict:
    """Compute various common metrics from the OOD detector scores:
    AUROC, FPR95TPR (or any other similar metric relative to confusion matrix),
    Detection accuracy and sklearn.metric metrics

    Args:
        scores (Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]): scores output of
            the OOD detector to evaluate. If a tuple is provided,
            the first array is considered in-distribution scores, and the second
            is considered out-of-distribution scores.
        labels (Optional[np.ndarray], optional): labels denoting oodness. When scores is
            a tuple, this argument and the following in_value and out_value are not
            used. If scores is a np.ndarray, labels are required with in_value and
            out_value if different from their default values.
            Defaults to None.
        in_value (Optional[int], optional): ood label value for in-distribution data.
            Defaults to 0.
        out_value (Optional[int], optional): ood label value for out-of-distribution
            data. Defaults to 1.
        metrics (Optional[List[str]], optional): list of metrics to compute. Can pass
            any metric name from sklearn.metric or among "detect_acc" and
            "<aaa><XX><bbb>" where <aaa> and <bbb> are in ["fpr", "tpr", "fnr", "tnr"]
            and <XX> is an integer between 1 and 99. Defaults to ["auroc", "fpr95tpr"].
        threshold (Optional[float], optional): Threshold to use when using
            threshold-dependent metrics. Defaults to None.
        step (Optional[int], optional): integration step (wrt percentile).
            Only used for auroc and fpr95tpr. Defaults to 4.

    Returns:
        dict: Dictionnary of metrics
    """
    metrics_dict = {}

    if isinstance(scores, np.ndarray):
        assert labels is not None, (
            "Provide labels with scores, or provide a tuple of in-distribution "
            "and out-of-distribution scores arrays"
        )
        labels = np.copy(labels)  # to avoid mutable np.array to be modified
        labels[labels == in_value] = 0
        labels[labels == out_value] = 1
    elif isinstance(scores, tuple):
        scores_in, scores_out = scores
        scores = np.concatenate([scores_in, scores_out])
        labels = np.concatenate([scores_in * 0, scores_out * 0 + 1])

    fpr, tpr, fnr, tnr, acc = get_curve(scores, labels, step)

    for metric in metrics:
        if isinstance(metric, str):
            if metric == "auroc":
                auroc = -np.trapz(1.0 - fpr, tpr)
                metrics_dict["auroc"] = auroc

            elif metric == "detect_acc":
                metrics_dict["detect_acc"] = np.max(acc)

            # compute <aaa><XX><bbb> metrics (check docstring for more info)
            elif (
                re.search(r"^(fpr|tpr|fnr|tnr)(\d{1,2})(fpr|tpr|fnr|tnr)$", metric)
                is not None
            ):
                count_1_str, thr, count_2_str = re.match(
                    pattern=r"^(fpr|tpr|fnr|tnr)(\d{1,2})(fpr|tpr|fnr|tnr)$",
                    string=metric,
                ).groups()
                thr = int(thr)
                count_1, count_2 = locals()[count_1_str], locals()[count_2_str]
                for i, c2 in enumerate(count_2):
                    if (count_2_str in ["fpr", "tpr"] and c2 < thr / 100) or (
                        count_2_str in ["tnr", "fnr"] and c2 > thr / 100
                    ):
                        ind = i
                        break
                metrics_dict[metric] = count_1[ind]

        elif metric.__name__ in sklearn.metrics.__all__:
            if metric.__name__[:3] == "roc":
                metrics_dict[metric.__name__] = metric(labels, scores)
            else:
                if threshold is None:
                    print(
                        f"No threshold is specified for metric {metric.__name__}, "
                        "skipping"
                    )
                else:
                    oodness = [1 if x > threshold else 0 for x in scores]
                    metrics_dict[metric.__name__] = metric(labels, oodness)

        else:
            print(f"Metric {metric.__name__} not implemented, skipping")

    return metrics_dict


def get_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    step: Optional[int] = 4,
    return_raw: Optional[bool] = False,
) -> Union[Tuple[tuple, tuple], tuple]:
    """Computes the
        * true positive rate: TP / (TP + FN),
        * false positive rate: FP / (FP + TN),
        * true negative rate: TN / (FP + TN),
        * false negative rate: FN / (TP + FN),
        * accuracy: (TN + TP) / (TP + FP + TN + FN),
    for different threshold values. The values are uniformly
    distributed among the percentiles, with a step = 4 / scores.shape[0]

    Args:
        scores (np.ndarray): scores output of the OOD detector to evaluate
        labels (np.ndarray): 1 if ood else 0
        step (Optional[int], optional): integration step (wrt percentile).
            Defaults to 4.
        return_raw (Optional[bool], optional): To return all the curves
            or only the rate curves. Defaults to False.

    Returns:
        Union[Tuple[Tuple[np.ndarray], Tuple[np.ndarray]], Tuple[np.ndarray]]: curves
    """
    tpc = np.array([])
    fpc = np.array([])
    tnc = np.array([])
    fnc = np.array([])
    thresholds = np.sort(scores)
    for i in range(1, len(scores), step):
        fp, tp, fn, tn = ftpn(scores, labels, thresholds[i])
        tpc = np.append(tpc, tp)
        fpc = np.append(fpc, fp)
        tnc = np.append(tnc, tn)
        fnc = np.append(fnc, fn)

    fpr = np.concatenate([[1.0], fpc / (fpc + tnc), [0.0]])
    tpr = np.concatenate([[1.0], tpc / (tpc + fnc), [0.0]])
    tnr = np.concatenate([[0.0], tnc / (fpc + tnc), [1.0]])
    fnr = np.concatenate([[0.0], fnc / (tpc + fnc), [1.0]])
    acc = (tnc + tpc) / (tpc + fpc + tnc + fnc)

    if return_raw:
        return (fpc, tpc, fnc, tnc), (fpr, tpr, fnr, tnr, acc)
    else:
        return fpr, tpr, fnr, tnr, acc


def ftpn(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple:
    """Computes the number of
        * true positives,
        * false positives,
        * true negatives,
        * false negatives,
    for a given threshold

    Args:
        scores (np.ndarray): scores output of the OOD detector to evaluate
        labels (np.ndarray): 1 if ood else 0
        threshold (float): threshold to use to consider scores
            as in-distribution or out-of-distribution

    Returns:
        Tuple[float]: The four metrics
    """
    pos = np.where(scores >= threshold)
    neg = np.where(scores < threshold)
    n_pos = len(pos[0])
    n_neg = len(neg[0])

    tp = np.sum(labels[pos])
    fp = n_pos - tp
    fn = np.sum(labels[neg])
    tn = n_neg - fn

    return fp, tp, fn, tn
