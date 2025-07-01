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
import pytest

from oodeel.aggregator import FisherAggregator
from oodeel.aggregator import MeanNormalizedAggregator
from oodeel.aggregator import StdNormalizedAggregator
from oodeel.methods import DKNN
from tests.tests_torch import eval_detector_on_blobs


@pytest.mark.parametrize(
    "auroc_thr,fpr95_thr,agg",
    [
        (0.95, 0.05, "fisher"),
        (0.95, 0.05, "mean"),
        (0.95, 0.05, "std"),
        (0.95, 0.05, "none"),
    ],
)
def test_dknn(auroc_thr, fpr95_thr, agg):
    """
    Test DKNN on toy blobs OOD dataset-wise task

    We check that the area under ROC is above a certain threshold, and that the FPR95TPR
    is below an other threshold.
    """
    aggregator = {
        "fisher": FisherAggregator,
        "mean": MeanNormalizedAggregator,
        "std": StdNormalizedAggregator,
        "none": lambda: None,
    }[agg]()
    dknn = DKNN(aggregator=aggregator)
    eval_detector_on_blobs(
        detector=dknn,
        auroc_thr=auroc_thr,
        fpr95_thr=fpr95_thr,
    )
