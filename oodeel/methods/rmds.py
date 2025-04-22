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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from ..aggregator import BaseAggregator
from ..aggregator import StdNormalizedAggregator
from ..types import DatasetType
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
        eps (float): Magnitude for gradient based input perturbation. Defaults to 0.002.
        aggregator (Optional[BaseAggregator]): Aggregator to combine scores from
            multiple feature layers. For a single layer this can be left as None.
    """

    def __init__(self, eps: float = 0.002, aggregator: Optional[BaseAggregator] = None):
        super().__init__(eps=eps, aggregator=aggregator)
        # Will be filled by `_fit_to_dataset`.
        self._layer_background_stats: List[Tuple[TensorType, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Per-layer helpers
    # ------------------------------------------------------------------

    def _fit_layer(
        self,
        layer_features: TensorType,
        labels: TensorType,
        subset_size: int = 5000,
    ) -> Tuple[
        Tuple[Dict[int, TensorType], TensorType, TensorType, TensorType],
        Optional[np.ndarray],
    ]:
        """Fit *one layer* and optionally return validation-subset scores.

        Returns
            layer_stats : (mus, pinv_cov, mu_bg, pinv_cov_bg)
                - `mus` - per-class means
                - `pinv_cov` - shared pseudo-inverse covariance
                - `mu_bg` - background mean
                - `pinv_cov_bg` - background pseudo-inverse covariance
            val_scores : np.ndarray | None
                Per-sample RMDS scores for an aggregator, or `None`.
        """
        mus, pinv_cov = super()._compute_layer_stats(layer_features, labels)
        mu_bg, pinv_cov_bg = self._background_stats(layer_features)

        scores: Optional[np.ndarray] = None
        if self.aggregator is not None:
            n_samples = min(subset_size, layer_features.shape[0])
            feats_subset = self.op.flatten(layer_features[:n_samples])
            g_scores = self._gaussian_log_probs(feats_subset, mus, pinv_cov)
            bg_score = self._background_log_prob(feats_subset, mu_bg, pinv_cov_bg)
            corrected = self.op.max(g_scores - bg_score, dim=1)
            scores = -self.op.convert_to_numpy(corrected)
        return (mus, pinv_cov, mu_bg, pinv_cov_bg), scores

    def _score_layer(
        self,
        out_features: TensorType,
        mus: Dict[int, TensorType],
        pinv_cov: TensorType,
        mu_bg: TensorType,
        pinv_cov_bg: TensorType,
    ) -> np.ndarray:
        """Compute the residual Mahalanobis OOD score for a single layer.

        Args:
            out_features (TensorType): Flattened feature matrix for the current
                batch (shape `[B, D]`).
            mus (Dict[int, TensorType]): Mapping class to mean for the layer.
            pinv_cov (TensorType): Shared pseudo-inverse covariance matrix.
            mu_bg (TensorType): Background mean vector.
            pinv_cov_bg (TensorType): Background pseudo-inverse covariance.

        Returns:
            np.ndarray: 1-D array (length `B`) of **negative** residual log-likelihoods.
                Higher values indicate a higher likelihood of the sample being
                out-of-distribution.
        """
        g_scores = self._gaussian_log_probs(out_features, mus, pinv_cov)
        bg_score = self._background_log_prob(out_features, mu_bg, pinv_cov_bg)
        corrected = self.op.max(g_scores - bg_score, dim=1)
        return -self.op.convert_to_numpy(corrected)

    # ==================================================================
    # Fit / score
    # ==================================================================

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Fit the RMDS detector using in-distribution data by computing
        both class-conditional statistics (per layer) and background statistics for each
        layer.

        For each feature layer, this method computes:
          - Class means and a weighted average covariance matrix (with its
            pseudo-inverse), stored in self._layer_stats.
          - The background mean and covariance matrix (with its pseudo-inverse),
            stored in self._layer_background_stats.

        If multiple layers are available and an aggregator is used, per-layer scores
        on a subset of samples are computed for fitting the aggregator.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): In-distribution dataset used
                for fitting the detector.

        """
        n_layers = len(self.feature_extractor.feature_layers_id)

        # Set default postprocessing functions if not provided.
        if self.postproc_fns is None:
            self.postproc_fns = [self.feature_extractor._default_postproc_fn] * n_layers

        # Extract features and labels
        features, infos = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, detach=True
        )
        labels = infos["labels"]

        self._layer_stats.clear()
        self._layer_background_stats.clear()
        per_layer_scores: List[np.ndarray] = []

        # Compute class-conditional statistics per layer.
        for i in range(n_layers):
            stats, scores = self._fit_layer(features[i], labels)
            mus, pinv_cov, mu_bg, pinv_cov_bg = stats
            self._layer_stats.append((mus, pinv_cov))
            self._layer_background_stats.append((mu_bg, pinv_cov_bg))
            if scores is not None:
                per_layer_scores.append(scores)

        if self.aggregator is None and n_layers > 1:
            self.aggregator = StdNormalizedAggregator()

        if self.aggregator is not None and per_layer_scores:
            self.aggregator.fit(per_layer_scores)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Compute OOD scores for input samples based on the RMDS distance.

        The process is as follows:
          1. Optionally apply gradient-based input perturbation.
          2. For each feature layer, extract features and compute:
             - The class-conditional Mahalanobis scores.
             - The background Mahalanobis scores.
          3. For each layer, subtract the background score from the class-conditional
             score, and select the maximum score among classes.
          4. When using multiple layers, aggregate the corrected per-layer scores via
             the provided aggregator.

        Args:
            inputs (TensorType): Input samples to be scored.

        Returns:
            Tuple[np.ndarray]: The final OOD scores (negative confidence values) for
                each sample.
        """
        x = self._input_perturbation(inputs) if self.eps > 0 else inputs
        features, _ = self.feature_extractor.predict_tensor(
            x, postproc_fns=self.postproc_fns
        )

        per_layer_scores = []
        for i in range(len(self._layer_stats)):
            out_f = self.op.flatten(features[i])
            mus, pinv_cov = self._layer_stats[i]
            mu_bg, pinv_cov_bg = self._layer_background_stats[i]
            per_layer_scores.append(
                self._score_layer(out_f, mus, pinv_cov, mu_bg, pinv_cov_bg)
            )

        if len(per_layer_scores) > 1 and self.aggregator is not None:
            return self.aggregator.aggregate(per_layer_scores)
        return per_layer_scores[0]
