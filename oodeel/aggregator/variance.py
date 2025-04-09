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


class VarianceNormalizedAggregator(BaseAggregator):
    """
    Aggregator that normalizes per-layer scores by their standard deviation before
    aggregating them.

    This aggregator computes the standard deviation of the OOD detection scores for
    each feature layer during the fitting stage. In the aggregation stage, each layer's
    score is normalized by its respective standard deviation, and then the average of
    the normalized scores is computed to produce the final score.
    """

    def __init__(self):
        self.stds = None

    def fit(self, per_layer_scores: List[np.ndarray]) -> None:
        """
        Compute and store the standard deviation for each feature layer's scores from
        training data.

        The standard deviation is calculated for each layer across all training samples.
        A small epsilon (1e-10) is added to each standard deviation to safeguard
        against division by zero during normalization.

        Args:
            per_layer_scores (List[np.ndarray]): A list of numpy arrays where each array
                contains the scores for a specific feature layer
                (shape: (num_samples,)).
        """
        # Stack scores such that the resulting shape is (num_samples, num_layers)
        scores_stack = np.stack(per_layer_scores, axis=-1)
        # Compute standard deviation per layer and add a small epsilon to avoid division
        # by zero
        self.stds = scores_stack.std(axis=0, keepdims=True) + 1e-10

    def aggregate(self, per_layer_scores: List[np.ndarray]) -> np.ndarray:
        """
        Normalize each feature layer's scores by its standard deviation and average
        across layers.

        Each per-layer score is divided by the corresponding precomputed standard
        deviation. The final aggregated score for each sample is the mean of the
        normalized scores across all layers.

        Args:
            per_layer_scores (List[np.ndarray]): A list of numpy arrays with scores from
                different feature layers. Each array should be of shape (num_samples,).

        Returns:
            np.ndarray: A 1D numpy array containing the aggregated score for each
                sample.
        """
        scores_stack = np.stack(per_layer_scores, axis=-1)
        if self.stds is None:
            raise ValueError("Aggregator has not been fitted yet.")
        # Normalize per layer and compute the mean over layers
        normalized_scores = scores_stack / self.stds
        return normalized_scores.mean(axis=-1)
