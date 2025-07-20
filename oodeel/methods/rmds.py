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
from typing import Optional
from typing import Tuple

import numpy as np

from ..aggregator import BaseAggregator
from ..types import TensorType
from .mahalanobis import Mahalanobis


class RMDS(Mahalanobis):
    """
    "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection"
    Updated to work with multiple feature layers.

    This detector computes class-conditional Mahalanobis scores from several
    feature layers and, additionally, computes background Mahalanobis scores per layer.
    The final per-layer score is obtained by subtracting the background score from the
    class-conditional score. With multiple layers, the per-layer scores are aggregated
    by a provided aggregator (or by a default StdNormalizedAggregator if None is
    given).

    Args:
        eps (float): Perturbation noise. Defaults to 0.0014.
        temperature (float, optional): Temperature parameter. Defaults to 1000.
        aggregator (Optional[BaseAggregator]): Aggregator to combine scores from
            multiple feature layers. For a single layer this can be left as None.
    """

    def __init__(
        self,
        eps: float = 0.0014,
        temperature: float = 1000,
        aggregator: Optional[BaseAggregator] = None,
        **kwargs,
    ):
        super().__init__(
            eps=eps, temperature=temperature, aggregator=aggregator, **kwargs
        )
        # Will be filled by `_fit_layer`.
        self._layer_background_stats: List[Tuple[TensorType, np.ndarray]] = []

    # === Per-layer logic ===
    def _fit_layer(
        self,
        layer_id: int,
        layer_features: np.ndarray,
        info: dict,
        **kwargs,
    ) -> None:
        """Compute statistics for a single layer and store parameters.

        Args:
            layer_id: Index of the processed layer. Unused here.
            layer_features: In-distribution features for this layer.
            info: Dictionary containing the training labels.
        """
        labels = info["labels"]

        if isinstance(layer_features, np.ndarray):
            layer_features = self.op.from_numpy(layer_features)

        mus, pinv_cov = super()._compute_layer_stats(layer_features, labels)
        mu_bg, pinv_cov_bg = self._background_stats(layer_features)

        self._layer_stats.append((mus, pinv_cov))
        self._layer_background_stats.append((mu_bg, pinv_cov_bg))

    def _score_layer(
        self,
        layer_id: int,
        layer_features: TensorType,
        info: dict,
        fit: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the residual Mahalanobis OOD score for a single layer.

        Args:
            layer_id: Index of the processed layer.
            layer_features: Flattened feature matrix `[B, D]` for the batch.
            info: Unused dictionary of auxiliary data.
            fit: Whether scoring is performed during fitting. Unused here.

        Returns:
            np.ndarray: 1-D array of **negative** residual log-likelihoods.
        """
        mus, pinv_cov = self._layer_stats[layer_id]
        mu_bg, pinv_cov_bg = self._layer_background_stats[layer_id]
        feats = self.op.flatten(layer_features)
        g_scores = self._gaussian_log_probs(feats, mus, pinv_cov)
        bg_score = self._background_log_prob(feats, mu_bg, pinv_cov_bg)
        corrected = self.op.max(g_scores - bg_score, dim=1)
        return -self.op.convert_to_numpy(corrected)

    # === Internal utilities ===
    def _background_stats(
        self, layer_features: TensorType
    ) -> Tuple[TensorType, TensorType]:
        """Compute class-agnostic Gaussian statistics for **one** layer.

        This helper forms the *background* distribution used in RMDS.  It treats
        **all** in-distribution samples of a layer as coming from a single
        multivariate Gaussian and returns its mean and (pseudo-inverse)
        covariance.

        Args:
            layer_features (TensorType): Feature representations of shape
                `[N, ...]` for a single layer, where `N` is the number of
                in-distribution training samples.

        Returns:
            Tuple[TensorType, TensorType]:
                * `mu_bg` - mean feature vector of shape `[D]` (same backend
                  tensor type as the inputs).
                * `pinv_cov_bg` - pseudo-inverse covariance matrix of shape
                  `[D, D]`.
        """
        feats = self.op.flatten(layer_features)
        mu_bg = self.op.mean(feats, dim=0)
        zero = feats - mu_bg
        cov_bg = self.op.matmul(self.op.t(zero), zero) / zero.shape[0]
        pinv_cov_bg = self.op.pinv(cov_bg)
        return mu_bg, pinv_cov_bg

    def _background_log_prob(
        self, out_features: TensorType, mu_bg: TensorType, pinv_cov_bg: TensorType
    ) -> TensorType:
        """
        Compute the Mahalanobis-based background score for a single feature layer.

        For each test sample, this method computes the log probability (up to a
        constant) under the background Gaussian distribution estimated from the
        in-distribution data.

        Args:
            out_features (TensorType): Feature tensor for test samples.
            mu_bg: Background mean vector for the layer.
            pinv_cov_bg (TensorType): Pseudo-inverse of the background covariance
                matrix.

        Returns:
            TensorType: Background confidence scores (reshaped as [num_samples, 1]).
        """
        zero = out_features - mu_bg
        log_prob = -0.5 * self.op.diag(
            self.op.matmul(self.op.matmul(zero, pinv_cov_bg), self.op.t(zero))
        )
        return self.op.reshape(log_prob, (-1, 1))
