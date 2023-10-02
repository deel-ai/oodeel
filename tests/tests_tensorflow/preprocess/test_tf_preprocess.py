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
import tensorflow as tf

from oodeel.preprocess import TFRandomPatchPermutation
from oodeel.types import Tuple
from tests.tests_tensorflow.tools_tf import almost_equal


@pytest.mark.parametrize(
    "patch_size, p_tensor_gt",
    [
        (
            (2, 2),
            np.array(
                [
                    [
                        [69.0, 70.0, 9.0, 10.0, 41.0, 42.0, 43.0, 44.0, 89.0, 90.0],
                        [79.0, 80.0, 19.0, 20.0, 51.0, 52.0, 53.0, 54.0, 99.0, 100.0],
                        [5.0, 6.0, 87.0, 88.0, 23.0, 24.0, 63.0, 64.0, 85.0, 86.0],
                        [15.0, 16.0, 97.0, 98.0, 33.0, 34.0, 73.0, 74.0, 95.0, 96.0],
                        [7.0, 8.0, 83.0, 84.0, 27.0, 28.0, 1.0, 2.0, 81.0, 82.0],
                        [17.0, 18.0, 93.0, 94.0, 37.0, 38.0, 11.0, 12.0, 91.0, 92.0],
                        [45.0, 46.0, 67.0, 68.0, 47.0, 48.0, 25.0, 26.0, 21.0, 22.0],
                        [55.0, 56.0, 77.0, 78.0, 57.0, 58.0, 35.0, 36.0, 31.0, 32.0],
                        [65.0, 66.0, 49.0, 50.0, 29.0, 30.0, 3.0, 4.0, 61.0, 62.0],
                        [75.0, 76.0, 59.0, 60.0, 39.0, 40.0, 13.0, 14.0, 71.0, 72.0],
                    ],
                    [
                        [69.0, 70.0, 9.0, 10.0, 41.0, 42.0, 43.0, 44.0, 89.0, 90.0],
                        [79.0, 80.0, 19.0, 20.0, 51.0, 52.0, 53.0, 54.0, 99.0, 100.0],
                        [5.0, 6.0, 87.0, 88.0, 23.0, 24.0, 63.0, 64.0, 85.0, 86.0],
                        [15.0, 16.0, 97.0, 98.0, 33.0, 34.0, 73.0, 74.0, 95.0, 96.0],
                        [7.0, 8.0, 83.0, 84.0, 27.0, 28.0, 1.0, 2.0, 81.0, 82.0],
                        [17.0, 18.0, 93.0, 94.0, 37.0, 38.0, 11.0, 12.0, 91.0, 92.0],
                        [45.0, 46.0, 67.0, 68.0, 47.0, 48.0, 25.0, 26.0, 21.0, 22.0],
                        [55.0, 56.0, 77.0, 78.0, 57.0, 58.0, 35.0, 36.0, 31.0, 32.0],
                        [65.0, 66.0, 49.0, 50.0, 29.0, 30.0, 3.0, 4.0, 61.0, 62.0],
                        [75.0, 76.0, 59.0, 60.0, 39.0, 40.0, 13.0, 14.0, 71.0, 72.0],
                    ],
                ]
            ),
        ),
        (
            (4, 3),
            np.array(
                [
                    [
                        [41.0, 42.0, 43.0, 7.0, 8.0, 9.0, 47.0, 48.0, 49.0, 0.0],
                        [51.0, 52.0, 53.0, 17.0, 18.0, 19.0, 57.0, 58.0, 59.0, 0.0],
                        [61.0, 62.0, 63.0, 27.0, 28.0, 29.0, 67.0, 68.0, 69.0, 0.0],
                        [71.0, 72.0, 73.0, 37.0, 38.0, 39.0, 77.0, 78.0, 79.0, 0.0],
                        [44.0, 45.0, 46.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
                        [54.0, 55.0, 56.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0],
                        [64.0, 65.0, 66.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 0.0],
                        [74.0, 75.0, 76.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [41.0, 42.0, 43.0, 7.0, 8.0, 9.0, 47.0, 48.0, 49.0, 0.0],
                        [51.0, 52.0, 53.0, 17.0, 18.0, 19.0, 57.0, 58.0, 59.0, 0.0],
                        [61.0, 62.0, 63.0, 27.0, 28.0, 29.0, 67.0, 68.0, 69.0, 0.0],
                        [71.0, 72.0, 73.0, 37.0, 38.0, 39.0, 77.0, 78.0, 79.0, 0.0],
                        [44.0, 45.0, 46.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
                        [54.0, 55.0, 56.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0],
                        [64.0, 65.0, 66.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 0.0],
                        [74.0, 75.0, 76.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ),
    ],
)
def test_random_patch_permutation(patch_size: Tuple[int], p_tensor_gt: np.ndarray):
    h, w, c = 10, 10, 2
    dummy_image = (
        np.arange(1, 101).reshape((h, w, 1)).repeat(c, axis=-1).astype(np.float32)
    )
    tensor = tf.convert_to_tensor(dummy_image)
    transform = TFRandomPatchPermutation(patch_size=patch_size)
    p_tensor = transform(tensor, seed=0)  # set seed for deterministic behaviour

    # shape assert
    assert p_tensor.shape == tensor.shape
    # compare tensors
    p_tensor_gt = p_tensor_gt.transpose(1, 2, 0)
    assert almost_equal(p_tensor.numpy(), p_tensor_gt)
