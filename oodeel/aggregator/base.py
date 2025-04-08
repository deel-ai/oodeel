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


class BaseAggregator:
    """
    Base class for aggregating out-of-distribution (OOD) detection scores from multiple
    feature layers.

    This abstract class defines the interface for aggregators that combine per-layer
    scores into a single score per sample. Subclasses should implement the `fit` and
    `aggregate` methods to capture any necessary statistics from training data
    and to combine scores during inference.
    """

    def fit(self, per_layer_scores: List[np.ndarray]) -> None:
        """
        Fit the aggregator on per-layer scores computed from in-distribution (ID)
        training data.

        This method extracts any statistical properties (e.g., standard deviations)
        from the provided scores that will be used later to normalize or weight the
        per-layer scores during aggregation.

        Args:
            per_layer_scores (List[np.ndarray]): A list of numpy arrays, where each
                array contains the scores for a particular feature layer
                (expected shape: (num_samples,)).
        """
        pass

    def aggregate(self, per_layer_scores: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate scores from multiple feature layers into a single score for each
        sample.

        This method should combine the per-layer scores (e.g., via normalization and
        averaging) into a unified score that can be used for OOD detection.

        Args:
            per_layer_scores (List[np.ndarray]): A list of numpy arrays,
                representing the scores from one feature layer for all samples.

        Returns:
            np.ndarray: An array containing the aggregated score for each sample.
        """
        raise NotImplementedError("Aggregator must implement an aggregate method.")
