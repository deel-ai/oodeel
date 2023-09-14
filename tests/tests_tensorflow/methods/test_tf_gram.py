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
from oodeel.methods import Gram
from tests.tests_tensorflow import generate_data_tf
from tests.tests_tensorflow import generate_model


def test_gram_shape():
    """
    Test Mahalanobis on MNIST vs FashionMNIST OOD dataset-wise task

    We check that the area under ROC is above a certain threshold, and that the FPR95TPR
    is below an other threshold.
    """
    gram = Gram(orders=range(1, 6))

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tf(
        x_shape=input_shape, num_labels=num_labels, samples=samples
    ).batch(samples // 2)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    gram.fit(model, data, feature_layers_id=[-5, -2])
    score, _ = gram.score(data)
    assert score.shape == (100,)
    assert gram.min_maxs[0][0].shape == (5, 4, 2)
    assert gram.min_maxs[0][1].shape == (5, 1, 2)

    gram.fit(model, data, feature_layers_id=[-2])
    score, _ = gram.score(data)
    assert score.shape == (100,)
    assert gram.min_maxs[0][0].shape == (5, 1, 2)
