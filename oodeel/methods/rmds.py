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
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from ..aggregator import BaseAggregator
from ..aggregator import StdNormalizedAggregator
from ..types import DatasetType
from ..types import TensorType
from oodeel.methods.mahalanobis import Mahalanobis


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

    def __init__(
        self, eps: float = 0.002, aggregator: Optional[BaseAggregator] = None
    ) -> None:
        super().__init__(eps=eps, aggregator=aggregator)

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Fit the RMDS detector using in-distribution data by computing
        both class-conditional statistics (per layer) and background statistics for each
        layer.

        For each feature layer, this method computes:
          - Class means and a weighted average covariance matrix (with its
            pseudo-inverse),
            stored in self._layer_stats.
          - The background mean and covariance matrix (with its pseudo-inverse),
            stored in self._layer_background_stats.

        If multiple layers are available and an aggregator is used, per-layer scores
        on a subset of samples are computed for fitting the aggregator.
        """
        num_feature_layers = len(self.feature_extractor.feature_layers_id)
        # Set default postprocessing functions if not provided.
        if self.postproc_fns is None:
            self.postproc_fns = [
                self.feature_extractor._default_postproc_fn
            ] * num_feature_layers

        # Extract features and labels using the provided postprocessing functions.
        features, infos = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, detach=True
        )
        labels = infos["labels"]

        self._layer_stats = []
        # Compute class-conditional statistics per layer.
        for i in range(num_feature_layers):
            mus, pinv_cov = self._compute_layer_stats(features[i], labels)
            self._layer_stats.append((mus, pinv_cov))

        # Compute per-layer background statistics.
        self._layer_background_stats = []
        for i in range(num_feature_layers):
            layer_features = self.op.flatten(features[i])
            mu_bg = self.op.mean(layer_features, dim=0)
            zero_f_bg = layer_features - mu_bg
            cov_bg = (
                self.op.matmul(self.op.t(zero_f_bg), zero_f_bg) / zero_f_bg.shape[0]
            )
            pinv_cov_bg = self.op.pinv(cov_bg)
            self._layer_background_stats.append((mu_bg, pinv_cov_bg))

        # If there is more than one feature layer, ensure an aggregator is defined.
        if self.aggregator is None and num_feature_layers > 1:
            self.aggregator = StdNormalizedAggregator()

        # If an aggregator is provided, compute fit scores on the first few thousand
        # samples per layer (arbitrary limit to avoid excessive memory usage).
        if self.aggregator is not None:
            per_layer_scores = []
            num_samples = min(5000, features[0].shape[0])
            for i in range(num_feature_layers):
                out_features = self.op.flatten(features[i][:num_samples])
                mus, pinv_cov = self._layer_stats[i]
                # Compute class-conditional scores.
                gaussian_score = self._mahalanobis_score_layer(
                    out_features, mus, pinv_cov
                )
                # Compute background scores for the layer.
                bg_mu, pinv_cov_bg = self._layer_background_stats[i]
                background_score = self._background_score_layer(
                    out_features, bg_mu, pinv_cov_bg
                )
                # The corrected score per layer (higher values indicate a higher chance
                # of OOD).
                max_score = self.op.max(gaussian_score - background_score, dim=1)
                per_layer_scores.append(-self.op.convert_to_numpy(max_score))
            self.aggregator.fit(per_layer_scores)

    def _background_score_layer(
        self, out_features: TensorType, mu_bg, pinv_cov_bg
    ) -> TensorType:
        """
        Compute the Mahalanobis-based background score for a single feature layer.

        For each test sample, this method computes the log probability (up to a
        constant) under the background Gaussian distribution estimated from the
        in-distribution data.

        Args:
            out_features (TensorType): Feature tensor for test samples.
            mu_bg: Background mean vector for the layer.
            pinv_cov_bg (np.ndarray): Pseudo-inverse of the background covariance
                matrix.

        Returns:
            TensorType: Background confidence scores (reshaped as [num_samples, 1]).
        """
        zero_f = out_features - mu_bg
        log_probs_f = -0.5 * self.op.diag(
            self.op.matmul(self.op.matmul(zero_f, pinv_cov_bg), self.op.t(zero_f))
        )
        return self.op.reshape(log_probs_f, (-1, 1))

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
        if self.eps > 0:
            inputs_p = self._input_perturbation(inputs)
        else:
            inputs_p = inputs

        features, _ = self.feature_extractor.predict_tensor(
            inputs_p, postproc_fns=self.postproc_fns
        )
        scores = []
        num_feature_layers = len(self._layer_stats)
        for i in range(num_feature_layers):
            out_features = self.op.flatten(features[i])
            mus, pinv_cov = self._layer_stats[i]
            gaussian_score = self._mahalanobis_score_layer(out_features, mus, pinv_cov)
            bg_mu, pinv_cov_bg = self._layer_background_stats[i]
            background_score = self._background_score_layer(
                out_features, bg_mu, pinv_cov_bg
            )
            max_score = self.op.max(gaussian_score - background_score, dim=1)
            scores.append(-self.op.convert_to_numpy(max_score))
        if num_feature_layers > 1:
            aggregated_scores = self.aggregator.aggregate(scores)  # type: ignore
        else:
            aggregated_scores = scores[0]
        return aggregated_scores
