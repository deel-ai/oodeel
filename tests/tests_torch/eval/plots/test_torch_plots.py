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
import torch

from oodeel.eval.plots import plot_2D_features
from oodeel.eval.plots import plot_3D_features
from oodeel.eval.plots import plot_ood_scores
from oodeel.eval.plots import plot_roc_curve
from oodeel.eval.plots import plotly_3D_features
from oodeel.methods import MLS
from tests.tests_torch.torch_methods_utils import load_blob_mlp
from tests.tests_torch.torch_methods_utils import load_blobs_data


def test_mls_blobs_plots():
    # seed
    torch.manual_seed(1)

    # load data
    batch_size = 128
    ds_fit, ds_in, ds_out = load_blobs_data(batch_size)

    # get classifier
    model = load_blob_mlp()

    # fit ood detector
    detector = MLS()
    detector.fit(model)

    # ood scores
    scores_in = detector.score(ds_in)
    scores_out = detector.score(ds_out)

    # static plots (matplotlib + seaborn)
    plt.figure()
    plt.subplot(141)
    plot_2D_features(model, ds_in, -2, ds_out, "TSNE")
    plt.subplot(142, projection="3d")
    plot_3D_features(model, ds_in, -2, ds_out, "PCA")
    plt.subplot(143)
    plot_ood_scores(scores_in, scores_out)
    plt.subplot(144)
    plot_roc_curve(scores_in, scores_out)
    plt.show()

    # interactive plot (plotly)
    plotly_3D_features(model, ds_in, -2, ds_out, "PCA")
