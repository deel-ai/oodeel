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
from typing import List

import numpy as np

from .base import BaseAggregator


class MeanNormalizedAggregator(BaseAggregator):
    """
    Aggregator that normalizes per-layer scores by their mean before aggregating them.

    This aggregator mimics the behavior of the original Gram detector:
    during the fitting phase, it computes, for each layer, a normalization constant that
    is the average (mean) of the deviation scores (computed on a validation set). At
    test time, each layer's score is divided by its corresponding mean, and the final
    score is obtained by averaging across layers.
    """

    def __init__(self):
        self.means = None

    def fit(self, per_layer_scores: List[np.ndarray]) -> None:
        """
        Computes and stores the mean for each feature layer's scores.

        Args:
            per_layer_scores (List[np.ndarray]): A list of arrays where each array
                contains the per-layer scores (shape: (num_samples,)).
        """
        scores_stack = np.stack(
            per_layer_scores, axis=-1
        )  # shape: (num_samples, num_layers)
        self.means = scores_stack.mean(axis=0, keepdims=True) + 1e-10

    def aggregate(self, per_layer_scores: List[np.ndarray]) -> np.ndarray:
        """
        Normalizes each layer's scores by its mean and averages them.

        Args:
            per_layer_scores (List[np.ndarray]): A list of arrays with scores from
                different feature layers.

        Returns:
            np.ndarray: A 1D array of aggregated scores.
        """
        scores_stack = np.stack(per_layer_scores, axis=-1)
        if self.means is None:
            raise ValueError("Aggregator has not been fitted yet.")
        normalized_scores = scores_stack / self.means
        return normalized_scores.mean(axis=-1)
