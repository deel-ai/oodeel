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

from ..types import TensorType
from .base import OODBaseDetector


class Entropy(OODBaseDetector):
    r"""
    Entropy OOD score


    The method consists in using the Entropy of the input data computed using the Entropy
    $\sum_{c=0}^C p(y=c| x) \times log(p(y=c | x))$ where
    $p(y=c| x) = \text{model}(x)$.

    **Reference**
    https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html,
    Neurips 2019.

    """

    def __init__(self):
        super().__init__(output_layers_id=[-1])

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        entropy.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        # compute logits (softmax(logits,axis=1) is the actual softmax
        # output minimized using binary cross entropy)
        logits = self.feature_extractor(inputs)
        probits = self.op.softmax(logits)
        probits = self.op.convert_to_numpy(probits)
        scores = np.sum(probits * np.log(probits), axis=1)
        return -scores