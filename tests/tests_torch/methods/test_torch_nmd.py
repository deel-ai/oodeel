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

from oodeel.methods import NeuralMeanDiscrepancy
from tests.tests_torch import eval_detector_on_blobs
from tests.tests_torch import load_blob_mlp
from tests.tests_torch import load_blobs_data


def test_nmd_shape():
    """
    Test Neural Mean Discrepancy execution
    """

    # load data
    ds_fit, ds_in, ds_out = load_blobs_data()

    # get classifier
    model = load_blob_mlp()
    nmd = NeuralMeanDiscrepancy()
    nmd.fit(model, fit_dataset=ds_in, ood_dataset=ds_out, feature_layers_id=[-3, -2])

    scores_in, info_in = nmd.score(ds_in)
    scores_out, info_out = nmd.score(ds_out)
    assert scores_in.shape == (1028,)
    assert info_in["labels"].shape == (1028,)
    assert info_in["logits"].shape == (1028, 2)
    assert scores_out.shape == (972,)
    assert info_out["labels"].shape == (972,)
    assert info_out["logits"].shape == (972, 2)
