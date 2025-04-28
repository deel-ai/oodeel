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
from typing import Tuple

import numpy as np
import scipy.stats

from .base import BaseAggregator


class FisherAggregator(BaseAggregator):
    """
    Aggregator that combines out-of-distribution (OOD) scores from multiple feature
    layers using Fisher's method with Brown's correction.

    The aggregator operates in two phases:

    1. **Fitting:**
       It fits an empirical cumulative distribution function (ECDF) on in-distribution
       (ID) training or validation scores obtained from all feature layers. It then
       computes the Fisher combined test statistic from the ID scores and derives
       Brown's correction parameters (mean and variance of the Fisher scores),
       which are used to adjust for correlations between layers.

    2. **Aggregation:**
       At test time, given per-layer OOD scores, it computes p-values using the ECDF,
       combines these p-values with Fisher's method, applies Brown's correction, and
       returns an aggregated score indicating the likelihood that a sample is
       out-of-distribution. In this setting, higher aggregated scores correspond to a
       higher OOD likelihood.

    Methods:
        fit(per_layer_scores):
            Fit the aggregator using ID scores from each feature layer.
        aggregate(per_layer_scores):
            Compute and return an aggregated OOD score from per-layer scores for test
            samples.
    """

    def __init__(self) -> None:
        self.id_scores = None  # Stacked ID scores from training data (used for ECDF)
        self.y_ecdf = None  # Empirical CDF values corresponding to the ID scores
        self.id_fisher_scores = (
            None  # Fisher combined test statistic computed on the ID scores
        )
        self.mu = None  # Mean of the Fisher scores (for Brown's correction)
        self.sigma2 = None  # Variance of the Fisher scores (for Brown's correction)
        self.c = None  # Correction factor derived from sigma2 and mu
        self.kprime = None  # Effective degrees of freedom after Brown's correction

    def fit(self, per_layer_scores: List[np.ndarray]) -> None:
        """
        Fit the aggregator using in-distribution (ID) scores.

        This method performs the following steps:
          1. Stacks the per-layer ID scores.
          2. Computes the empirical CDF over the stacked scores.
          3. Computes the Fisher combined test statistic for the training scores.
          4. Derives Brown's correction parameters based on the mean and variance of the
            Fisher scores.

        Args:
            per_layer_scores (List[np.ndarray]): A list of 1D numpy arrays, where each
                array contains the OOD detection scores from a specific feature layer
                for the ID data.
        """
        # Stack scores so that the resulting shape is (num_samples, num_layers)
        id_scores = np.stack(per_layer_scores, axis=-1)
        # Compute empirical CDF based on the ID scores
        self.id_scores, self.y_ecdf = empirical_cdf(id_scores)
        # Compute Fisher's combined statistic for the ID scores
        self.id_fisher_scores = self._compute_fisher_scores(id_scores)
        # Derive Brown's correction parameters from the Fisher scores
        self.mu = np.mean(self.id_fisher_scores)
        self.sigma2 = np.var(self.id_fisher_scores)
        self.c = self.sigma2 / (2 * self.mu)
        self.kprime = 2 * self.mu**2 / self.sigma2

    def _compute_p_values(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute p-values for test scores based on the empirical CDF from the training
        data.

        Args:
            scores (np.ndarray): A numpy array of stacked test scores with shape
                (num_samples, num_layers).

        Returns:
            np.ndarray: A numpy array of p-values with the same shape as the input.
        """
        return p_value_fn(scores, self.id_scores, self.y_ecdf)

    def _compute_fisher_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute Fisher's combined test statistic for the given scores.

        This method first converts the scores into p-values (using the empirical CDF)
        and then computes the Fisher statistic for each sample by summing the logarithms
        of the p-values.

        Args:
            scores (np.ndarray): The stacked scores (ID or test) with shape
                (num_samples, num_layers).

        Returns:
            np.ndarray: A 1D array of Fisher combined test statistics, one for each
                sample.
        """
        p_values = self._compute_p_values(scores)
        return fisher_tau_method(p_values)

    def aggregate(self, per_layer_scores: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate per-layer scores into a single OOD score for each test sample.

        The aggregation process involves:
          1. Stacking the per-layer scores.
          2. Computing Fisher's combined test statistic.
          3. Applying Brown's correction to obtain adjusted p-values.
          4. Converting these p-values into an aggregated OOD score, where higher scores
            indicate a higher likelihood that a sample is out-of-distribution.

        Args:
            per_layer_scores (List[np.ndarray]): A list of 1D numpy arrays representing
                the OOD scores from each feature layer for the test data.

        Returns:
            np.ndarray: A 1D numpy array of aggregated OOD scores for each test sample.
        """
        scores = np.stack(per_layer_scores, axis=-1)
        fisher_scores = self._compute_fisher_scores(scores)
        # Apply Brown's correction: compute p-values from the corrected chi-square dist
        p_values = 1 - scipy.stats.chi2.cdf(fisher_scores / self.c, self.kprime)
        # Convert p-values to aggregated score (lower p-values => higher OOD likelihood)
        return 1 - p_values


def empirical_cdf(X: np.ndarray, w: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical cumulative distribution function (ECDF) for a given sample.

    The function first negates the input data (assuming that lower scores indicate
    higher in-distribution confidence), augments the sample with lower and upper bounds,
    sorts the values, and then computes the ECDF.

    Args:
        X (np.ndarray): An array of shape (N, m), where N is the number of samples and m
            is the number of feature layers.
        w (np.ndarray, optional): Optional weights to adjust the ECDF values. Defaults
            to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - An array of sorted (and augmented) sample values.
            - An array of ECDF values corresponding to the sample.
    """
    # Negate scores so that higher values correspond to higher in-dist confidence.
    X = -X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mult_factor_min = np.where(X.min(0) > 0, np.array(1 / len(X)), np.array(len(X)))
    mult_factor_max = np.where(X.max(0) > 0, np.array(len(X)), np.array(1 / len(X)))
    lower_bound = X.min(0) * mult_factor_min
    upper_bound = X.max(0) * mult_factor_max
    X_aug = np.concatenate(
        (lower_bound.reshape(1, -1), X, upper_bound.reshape(1, -1)), axis=0
    )
    X_sorted = np.sort(X_aug, axis=0)
    y_ecdf = np.concatenate(
        [np.arange(1, X_sorted.shape[0] + 1).reshape(-1, 1) / X_sorted.shape[0]]
        * X_sorted.shape[1],
        axis=1,
    )
    if w is not None:
        y_ecdf = y_ecdf * w.reshape(1, -1)
    return X_sorted, y_ecdf


def p_value_fn(
    test_statistic: np.ndarray, X: np.ndarray, y_ecdf: np.ndarray
) -> np.ndarray:
    """
    Compute p-values for the given test statistics using the empirical CDF.

    For each feature layer, this function linearly interpolates the test statistic
    values within the sorted training sample values (augmented with bounds) and returns
    the ECDF values.

    Args:
        test_statistic (np.ndarray): Array of test statistics with shape (n, m) where n
            is the number of test samples and m is the number of layers.
        X (np.ndarray): Sorted training sample values (with bounds) obtained from
            `empirical_cdf`, shape (N, m).
        y_ecdf (np.ndarray): Corresponding ECDF values for each layer, shape (N, m).

    Returns:
        np.ndarray: Interpolated p-values for the test samples with shape (n, m).
    """
    test_statistic = -test_statistic  # Ensure consistency with the ECDF computation
    interpolated = []
    for i in range(test_statistic.shape[1]):
        layer_test_stat = test_statistic[:, i]
        layer_X = X[:, i]
        layer_ecdf = y_ecdf[:, i]
        interp_values = np.interp(layer_test_stat, layer_X, layer_ecdf).reshape(-1, 1)
        interpolated.append(interp_values)
    return np.concatenate(interpolated, axis=1)


def fisher_tau_method(p_values: np.ndarray) -> np.ndarray:
    """
    Combine p-values using Fisher's method.

    For each sample, the Fisher statistic is computed as:
        tau = -2 * sum(log(p_i))
    where the sum is taken over the p-values from all feature layers.

    Args:
        p_values (np.ndarray): Array of p-values with shape (n, m).

    Returns:
        np.ndarray: A 1D array of Fisher combined statistics, one per test sample
            (shape: (n,)).
    """
    tau = -2 * np.sum(np.log(p_values), axis=1)
    return tau
