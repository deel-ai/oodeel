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
from torch.utils.data import DataLoader

from oodeel.methods import Energy
from tests.tests_torch import ComplexNet
from tests.tests_torch import generate_data
from tests.tests_torch import generate_data_torch


@pytest.mark.parametrize(
    ("backend", "input_shape"),
    [("torch", (3, 32, 32))],
    ids=["[torch] test DKNN"],
)
def test_dknn(backend, input_shape):
    """
    Test DKNN
    """
    num_labels = 10
    samples = 100

    data_x, _ = generate_data(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )

    data_x = generate_data_torch(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=True
    )
    data_x = DataLoader(data_x, batch_size=samples // 2)
    model = ComplexNet()

    dknn = Energy()
    dknn.fit(model)
    scores = dknn.score(data_x)

    assert scores.shape == (100,)
