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

from typing import Union

import numpy as np
import tensorflow as tf
from scipy.special import logsumexp

from .base import OODModel


class Energy(OODModel):
    r"""
    Energy Score method for OOD detection.
    "Energy-based Out-of-distribution Detection"
    https://arxiv.org/abs/2010.03759

    This method assumes that the model has been trained with cross entropy loss
    :math:'CE(model(x))' where :math:'model(x)=(l_{c})_{c=1}^{C}' are the logits
    predicted for input :math: 'x'.
    The implementation assumes that the logits are retreieved using the output with
    linear activation.

    The energy score for input :math:'x' is given by
    .. math:: -\log \sum_{c=0}^C \exp(l_c)

    where 'model(x)=(l_{c})_{c=1}^{C}' are the logits predicted by the model on
    :math:'x'.
    As always, training data is expected to have lower score than OOD data.


    Args:
        batch_size: batch_size used to compute the features space
            projection of input data.
            Defaults to 256.
    """

    def __init__(self):
        super().__init__(output_layers_id=[-1], input_layers_id=0)

    def _score_tensor(
        self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        energy, namey :math:'-logsumexp(logits(inputs))'.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"

        # compute logits (softmax(logits,axis=1) is the actual softmax
        # output minimized using binary cross entropy)
        logits = self.feature_extractor(inputs)
        scores = -logsumexp(logits, axis=1)
        return scores
