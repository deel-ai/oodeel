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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from oodeel.eval.metrics import bench_metrics
from oodeel.eval.metrics import get_curve

sns.set_style("darkgrid")


def plot_ood_scores(
    scores_in: np.ndarray,
    scores_out: np.ndarray,
    log_scale: bool = False,
    title: str = None,
):
    """Plot histograms of OOD detection scores for ID and OOD distribution, using
    matplotlib and seaborn.

    Args:
        scores_in (np.ndarray): OOD detection scores for ID data.
        scores_out (np.ndarray): OOD detection scores for OOD data.
        log_scale (bool, optional): If True, apply a log scale on x axis. Defaults to
            False.
        title (str, optional): Custom figure title. If None a default one is provided.
            Defaults to None.
    """
    title = title or "Histograms of OOD detection scores"
    ax1 = sns.histplot(
        data=scores_in,
        alpha=0.5,
        label="ID data",
        stat="density",
        log_scale=log_scale,
        kde=True,
    )
    ax2 = sns.histplot(
        data=scores_out,
        alpha=0.5,
        label="OOD data",
        stat="density",
        log_scale=log_scale,
        kde=True,
    )
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    threshold = np.percentile(scores_out, q=5.0)
    plt.vlines(
        x=[threshold],
        ymin=0,
        ymax=ymax,
        colors=["red"],
        linestyles=["dashed"],
        alpha=0.7,
        label="TPR=95%",
    )
    plt.xlabel("OOD score")
    plt.legend()
    plt.title(title, weight="bold").set_fontsize(11)


def plot_roc_curve(scores_in: np.ndarray, scores_out: np.ndarray, title: str = None):
    """Plot ROC curve for OOD detection task, using matplotlib and seaborn.

    Args:
        scores_in (np.ndarray): OOD detection scores for ID data.
        scores_out (np.ndarray): OOD detection scores for OOD data.
        title (str, optional): Custom figure title. If None a default one is provided.
            Defaults to None.
    """
    # compute auroc
    metrics = bench_metrics(
        (scores_in, scores_out),
        metrics=["auroc", "fpr95tpr"],
    )
    auroc, fpr95tpr = metrics["auroc"], metrics["fpr95tpr"]

    # roc
    fpr, tpr, _, _, _ = get_curve(
        scores=np.concatenate([scores_in, scores_out]),
        labels=np.concatenate([scores_in * 0 + 0, scores_out * 0 + 1]),
    )

    # plot roc
    title = title or "ROC curve (AuC = {:.3f})".format(auroc)
    plt.plot(fpr, tpr)
    plt.fill_between(fpr, tpr, np.zeros_like(tpr), alpha=0.5)
    plt.plot([fpr95tpr, fpr95tpr, 0], [0, 0.95, 0.95], "--", color="red", alpha=0.7)
    plt.scatter([fpr95tpr], [0.95], marker="o", alpha=0.7, color="red", label="TPR=95%")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend()
    plt.title(title, weight="bold").set_fontsize(11)
