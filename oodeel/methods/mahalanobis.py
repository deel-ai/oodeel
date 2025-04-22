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
from .base import OODBaseDetector


class Mahalanobis(OODBaseDetector):
    """
    "A Simple Unified Framework for Detecting Out-of-Distribution Samples and
    Adversarial Attacks"
    https://arxiv.org/abs/1807.03888

    This detector computes the Mahalanobis distance between the feature representations
    of input samples and class-conditional Gaussian distributions estimated from
    in-distribution data. It supports multiple feature layers by computing statistics
    (class means and a covariance matrix) for each layer. During inference, scores
    computed for each layer are aggregated using a provided aggregator.

    Args:
        eps (float): Magnitude for gradient-based input perturbation. Defaults to 0.002.
        aggregator (Optional[BaseAggregator]): Aggregator to combine scores from
            multiple feature layers. If `None` and more than one layer is
            used, a `StdNormalizedAggregator` is instantiated automatically.
    """

    def __init__(self, eps: float = 0.002, aggregator: Optional[BaseAggregator] = None):
        super().__init__()
        self.eps = eps
        self.aggregator = aggregator
        self._layer_stats: List[Tuple[Dict, np.ndarray]] = []
        self._classes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_layer_stats(
        self, layer_features: TensorType, labels: TensorType
    ) -> Tuple[Dict[int, TensorType], np.ndarray]:
        """
        Compute class-conditional statistics for a given feature layer.

        For each class present in the labels, this method computes the mean feature
        vector. It also computes a weighted average covariance matrix (across all
        classes) and its pseudo-inverse, which will later be used for computing
        Mahalanobis distances.

        Args:
            layer_features (TensorType): Feature tensor for a specific layer extracted
                from in-distribution data.
            labels (TensorType): Corresponding labels for the in-distribution data.

        Returns:
            Tuple[Dict, np.ndarray]:
                - A dictionary mapping each class label to its mean feature vector.
                - The pseudo-inverse of the weighted average covariance matrix.
        """
        labels_np = self.op.convert_to_numpy(labels)
        classes = np.sort(np.unique(labels_np))

        feats = self.op.flatten(layer_features)
        n_total = feats.shape[0]

        mus: Dict[int, TensorType] = {}
        mean_cov: Optional[np.ndarray] = None

        for cls in classes:
            idx = self.op.equal(labels, cls)
            feats_cls = self.op.flatten(layer_features[idx])
            mu = self.op.mean(feats_cls, dim=0)
            mus[cls] = mu

            zero_f = feats_cls - mu
            cov_cls = self.op.matmul(self.op.t(zero_f), zero_f) / zero_f.shape[0]
            weight = feats_cls.shape[0] / n_total
            mean_cov = (
                cov_cls * weight if mean_cov is None else mean_cov + cov_cls * weight
            )

        pinv_cov = self.op.pinv(mean_cov)  # type: ignore[arg-type]
        if self._classes is None:
            self._classes = classes
        return mus, pinv_cov

    def _gaussian_log_probs(
        self, out_features: TensorType, mus: Dict[int, TensorType], pinv_cov: np.ndarray
    ) -> TensorType:
        """Compute unnormalised Gaussian log-probabilities for all classes.

        Args:
            out_features (TensorType): Features of shape [B, D].
            mus (Dict[int, TensorType]): Class mean vectors.
            pinv_cov (np.ndarray): Pseudo-inverse covariance matrix.

        Returns:
            TensorType: Log-probabilities with shape [B, n_classes].
        """
        scores = []
        for cls in self._classes:  # type: ignore[assignment]
            mu = mus[cls]
            zero_f = out_features - mu
            log_prob = -0.5 * self.op.diag(
                self.op.matmul(self.op.matmul(zero_f, pinv_cov), self.op.t(zero_f))
            )
            scores.append(self.op.reshape(log_prob, (-1, 1)))
        return self.op.cat(scores, dim=1)

    def _input_perturbation(self, inputs: TensorType) -> TensorType:
        """Apply a small gradient-based perturbation (FGSM style) to the inputs to
        enhance the separation between in- and out-distribution samples.

        This method uses the first feature layer for computing the perturbation.
        See the original paper (section 2.2) for more details:
        https://arxiv.org/abs/1807.03888

        Args:
            inputs (TensorType): Input samples.

        Returns:
            TensorType: Perturbed input samples.
        """

        def _loss_fn(x: TensorType) -> TensorType:
            feats, _ = self.feature_extractor.predict(
                x, postproc_fns=self.postproc_fns, detach=False
            )
            out_f = self.op.flatten(feats[-1])
            mus, pinv_cov = self._layer_stats[-1]
            g_scores = self._gaussian_log_probs(out_f, mus, pinv_cov)
            max_score = self.op.max(g_scores, dim=1)
            return self.op.mean(-max_score)

        grad = self.op.gradient(_loss_fn, inputs)
        grad = self.op.sign(grad)
        return inputs - self.eps * grad

    # ------------------------------------------------------------------
    # Fit / score helpers
    # ------------------------------------------------------------------

    def _fit_layer(
        self, layer_features: TensorType, labels: TensorType, subset_size: int = 5000
    ) -> Tuple[Tuple[Dict[int, TensorType], np.ndarray], Optional[np.ndarray]]:
        """Fit statistics for a single layer and, if required, return scores.

        Args:
            layer_features (TensorType): In-distribution features for the layer.
            labels (TensorType): Class labels.
            subset_size (int, optional): Number of samples used to compute initial
                scores for the aggregator. Defaults to 5000.

        Returns:
            Tuple[Tuple[Dict[int, TensorType], np.ndarray], Optional[np.ndarray]]:
                * (mus, pinv_cov) statistics for the layer.
                * Optional numpy array of OOD scores for the first subset_size
                  samples (None if no aggregator).
        """
        mus, pinv_cov = self._compute_layer_stats(layer_features, labels)

        scores: Optional[np.ndarray] = None
        if self.aggregator is not None:
            n_samples = min(subset_size, layer_features.shape[0])
            feats_subset = self.op.flatten(layer_features[:n_samples])
            g_scores = self._gaussian_log_probs(feats_subset, mus, pinv_cov)
            max_scores = self.op.max(g_scores, dim=1)
            scores = -self.op.convert_to_numpy(max_scores)
        return (mus, pinv_cov), scores

    def _score_layer(
        self, out_features: TensorType, mus: Dict[int, TensorType], pinv_cov: np.ndarray
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance-based confidence scores for a single feature layer.

        For each test sample, this method computes the log probability density (up to a
        constant) under the Gaussian distribution corresponding to each class, using the
        Mahalanobis distance.

        Args:
            out_features (TensorType): Feature tensor for test samples.
            mus (Dict): Dictionary mapping each class label to its mean feature vector.
            pinv_cov (np.ndarray): Pseudo-inverse of the covariance matrix.

        Returns:
            TensorType: Confidence scores for each sample for every class,
                        with shape [num_samples, num_classes].
        """
        g_scores = self._gaussian_log_probs(out_features, mus, pinv_cov)
        max_score = self.op.max(g_scores, dim=1)
        return -self.op.convert_to_numpy(max_score)

    # ------------------------------------------------------------------
    # Fit / score
    # ------------------------------------------------------------------

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Fit the Mahalanobis detector using in-distribution data by computing
        class-conditional statistics for each feature layer.

        For each layer, the method calculates the class means and the weighted average
        covariance matrix, whose pseudo-inverse will be used for computing the
        Mahalanobis distances at test time. Additionally, if an aggregator is provided,
        it computes scores on the first 1000 samples for each layer and fits the
        aggregator to these scores (e.g. to compute standard deviations).

        Args:
            fit_dataset (Union[TensorType, DatasetType]): In-distribution data for
                fitting the detector.
        """
        n_layers = len(self.feature_extractor.feature_layers_id)

        if self.postproc_fns is None:
            self.postproc_fns = [self.feature_extractor._default_postproc_fn] * n_layers

        features, infos = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, detach=True
        )
        labels = infos["labels"]

        self._layer_stats.clear()
        per_layer_scores = []

        for i in range(n_layers):
            stats, scores = self._fit_layer(features[i], labels)
            self._layer_stats.append(stats)
            if scores is not None:
                per_layer_scores.append(scores)

        if self.aggregator is None and n_layers > 1:
            self.aggregator = StdNormalizedAggregator()

        if self.aggregator is not None and per_layer_scores:
            self.aggregator.fit(per_layer_scores)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Compute out-of-distribution scores for input samples based on the Mahalanobis
        distance.

        The procedure is as follows:
          1. (Optional) Apply input perturbation (if eps > 0) using the first feature
            layer.
          2. Extract features from the (possibly perturbed) inputs for each feature
            layer.
          3. For each layer, compute Mahalanobis-based scores by evaluating the log
            probability densities.
          4. For each layer, select the highest score across classes.
          5. If multiple layers are used, aggregate the per-layer scores via the
            provided aggregator.

        Args:
            inputs (TensorType): Input samples to be scored.

        Returns:
            Tuple[np.ndarray]: A tuple containing the final OOD scores for each input
                sample. The scores are returned as the negative of the computed
                confidence.
        """
        x = self._input_perturbation(inputs) if self.eps > 0 else inputs

        features, _ = self.feature_extractor.predict_tensor(
            x, postproc_fns=self.postproc_fns
        )
        n_layers = len(self._layer_stats)

        scores = []
        for i in range(n_layers):
            out_f = self.op.flatten(features[i])
            mus, pinv_cov = self._layer_stats[i]
            scores.append(self._score_layer(out_f, mus, pinv_cov))

        if n_layers > 1:
            return self.aggregator.aggregate(scores)  # type: ignore[arg-type]
        return scores[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Indicates whether this OOD detector requires in-distribution data for fitting.

        Returns:
            bool: True, since fitting requires computing class-conditional statistics.
        """
        return True

    @property
    def requires_internal_features(self) -> bool:
        """
        Indicates whether this OOD detector utilizes internal model features.

        Returns:
            bool: True, as it operates on intermediate feature representations.
        """
        return True
