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

import numpy as np

from ..aggregator import BaseAggregator
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

    # === Public API (override of _score_tensor) ===
    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """Compute Mahalanobis-based OOD scores for a batch of samples.

        The inputs may first be perturbed using the FGSM-style procedure
        introduced in the original paper. Features are then extracted for all
        configured layers and :func:`_score_layer` is called for each of them.
        If multiple layers are used, the per-layer scores are combined using an
        `aggregator` when available, otherwise the mean score across layers is
        returned.

        Args:
            inputs: Batch of samples to score.

        Returns:
            `np.ndarray` containing one score per input sample.
        """
        x = self._input_perturbation(inputs) if self.eps > 0 else inputs

        feats, logits = self.feature_extractor.predict_tensor(
            x, postproc_fns=self.postproc_fns
        )

        info = {"logits": logits}

        per_layer_scores = [
            self._score_layer(i, self.op.flatten(feats[i]), info)
            for i in range(len(self._layer_stats))
        ]

        if getattr(self, "aggregator", None) is not None and len(per_layer_scores) > 1:
            return self.aggregator.aggregate(per_layer_scores)  # type: ignore[arg-type]
        if len(per_layer_scores) > 1:
            return np.mean(np.stack(per_layer_scores, axis=1), axis=1)
        return per_layer_scores[0]

    # === Per-layer logic ===
    def _fit_layer(
        self,
        layer_id: int,
        layer_features: np.ndarray,
        info: dict,
        subset_size: int = 5000,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Compute class statistics for one feature layer.

        Args:
            layer_id: Index of the processed layer.
            layer_features: In-distribution features for this layer.
            info: Dictionary containing the training labels.
            subset_size: Number of samples used to compute preliminary scores
                for the aggregator.

        Returns:
            Optional[np.ndarray]: Scores on `subset_size` samples if an
            aggregator is used.
        """
        labels = info["labels"]

        if isinstance(layer_features, np.ndarray):
            layer_features = self.op.from_numpy(layer_features)

        mus, pinv_cov = self._compute_layer_stats(layer_features, labels)

        self._layer_stats.append((mus, pinv_cov))

        scores: Optional[np.ndarray] = None
        if getattr(self, "aggregator", None) is not None:
            n_samples = min(subset_size, layer_features.shape[0])
            feats_subset = self.op.flatten(layer_features[:n_samples])
            g_scores = self._gaussian_log_probs(feats_subset, mus, pinv_cov)
            max_scores = self.op.max(g_scores, dim=1)
            scores = -self.op.convert_to_numpy(max_scores)

        return scores

    def _score_layer(
        self,
        layer_id: int,
        out_features: TensorType,
        info: dict,
        **kwargs,
    ) -> np.ndarray:
        """Compute Mahalanobis confidence for one feature layer.

        Args:
            layer_id: Index of the processed layer.
            out_features: Feature tensor for the current batch.
            info: Unused dictionary of auxiliary data.

        Returns:
            np.ndarray: Negative Mahalanobis confidence scores.
        """
        mus, pinv_cov = self._layer_stats[layer_id]
        g_scores = self._gaussian_log_probs(out_features, mus, pinv_cov)
        max_score = self.op.max(g_scores, dim=1)
        return -self.op.convert_to_numpy(max_score)

    # === Internal utilities ===
    def _compute_layer_stats(
        self, layer_features: TensorType, labels: np.ndarray
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
            labels (np.ndarray): Corresponding labels for the in-distribution data.

        Returns:
            Tuple[Dict, TensorType]:
                - A dictionary mapping each class label to its mean feature vector.
                - The pseudo-inverse of the weighted average covariance matrix.
        """
        classes = np.sort(np.unique(labels))
        labels = self.op.from_numpy(labels)  # convert to tensor

        feats = self.op.flatten(layer_features)
        n_total = feats.shape[0]

        mus: Dict[int, TensorType] = {}
        mean_cov: TensorType = None

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
        self, out_features: TensorType, mus: Dict[int, TensorType], pinv_cov: TensorType
    ) -> TensorType:
        """Compute unnormalised Gaussian log-probabilities for all classes.

        Args:
            out_features (TensorType): Features of shape [B, D].
            mus (Dict[int, TensorType]): Class mean vectors.
            pinv_cov (TensorType): Pseudo-inverse covariance matrix.

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

    # ===
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
