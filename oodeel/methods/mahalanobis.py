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
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from ..aggregator import BaseAggregator
from ..aggregator import StdNormalizedAggregator
from ..types import DatasetType
from ..types import TensorType
from oodeel.methods.base import OODBaseDetector


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
            multiple feature layers. For single-layer use this can be left as None.
            If multiple layers are used, an aggregator must be provided.
    """

    def __init__(
        self, eps: float = 0.002, aggregator: Optional[BaseAggregator] = None
    ) -> None:
        super(Mahalanobis, self).__init__()
        self.eps: float = eps
        self.aggregator: Optional[BaseAggregator] = aggregator

    def _compute_layer_stats(
        self, layer_features: TensorType, labels: TensorType
    ) -> Tuple[Dict, np.ndarray]:
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

        features_np = self.op.flatten(layer_features)
        n_total = features_np.shape[0]

        mus = {}
        mean_cov = None

        for cls in classes:
            indexes = self.op.equal(labels, cls)
            features_cls = self.op.flatten(layer_features[indexes])
            mus[cls] = self.op.mean(features_cls, dim=0)
            zero_f = features_cls - mus[cls]
            cov_cls = self.op.matmul(self.op.t(zero_f), zero_f) / zero_f.shape[0]
            weight = features_cls.shape[0] / n_total
            if mean_cov is None:
                mean_cov = weight * cov_cls
            else:
                mean_cov += weight * cov_cls

        pinv_cov = self.op.pinv(mean_cov)
        if not hasattr(self, "_classes"):
            self._classes = classes
        return mus, pinv_cov

    def _mahalanobis_score_layer(
        self, out_features: TensorType, mus: Dict, pinv_cov: np.ndarray
    ) -> TensorType:
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
        gaussian_scores = []
        for cls in self._classes:
            mu = mus[cls]
            zero_f = out_features - mu
            log_probs_f = -0.5 * self.op.diag(
                self.op.matmul(self.op.matmul(zero_f, pinv_cov), self.op.t(zero_f))
            )
            gaussian_scores.append(self.op.reshape(log_probs_f, (-1, 1)))
        gaussian_score = self.op.cat(gaussian_scores, dim=1)
        return gaussian_score

    def _input_perturbation(self, inputs: TensorType) -> TensorType:
        """
        Apply a small gradient-based perturbation to the inputs to enhance the
        separation between in- and out-distribution samples.

        This method uses the first feature layer for computing the perturbation.
        See the original paper (section 2.2) for more details:
        https://arxiv.org/abs/1807.03888

        Args:
            inputs (TensorType): Input samples.

        Returns:
            TensorType: Perturbed input samples.
        """

        def __loss_fn(x: TensorType) -> TensorType:
            out_features, _ = self.feature_extractor.predict(
                x, postproc_fns=self.postproc_fns, detach=False
            )
            out_features_0 = self.op.flatten(out_features[-1])
            mus, pinv_cov = self._layer_stats[-1]
            gaussian_score = self._mahalanobis_score_layer(
                out_features_0, mus, pinv_cov
            )
            max_score = self.op.max(gaussian_score, dim=1)
            return self.op.mean(-max_score)

        gradient = self.op.gradient(__loss_fn, inputs)
        gradient = self.op.sign(gradient)
        inputs_p = inputs - self.eps * gradient
        return inputs_p

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
        num_feature_layers = len(self.feature_extractor.feature_layers_id)
        # Set default postprocessing functions if not provided.
        if self.postproc_fns is None:
            self.postproc_fns = [
                self.feature_extractor._default_postproc_fn
            ] * num_feature_layers

        features, infos = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, detach=True
        )
        labels = infos["labels"]

        self._layer_stats = []

        # Compute per-layer statistics
        for i in range(num_feature_layers):
            mus, pinv_cov = self._compute_layer_stats(features[i], labels)
            self._layer_stats.append((mus, pinv_cov))

        # If there is more than one feature layer, ensure an aggregator is defined.
        if self.aggregator is None and num_feature_layers > 1:
            self.aggregator = StdNormalizedAggregator()

        # If an aggregator is provided, compute aggregator fit scores on the first 1000
        # samples per layer.
        if self.aggregator is not None:
            per_layer_scores = []
            num_samples = min(5000, features[0].shape[0])
            for i in range(num_feature_layers):
                out_features = self.op.flatten(features[i][:num_samples])
                mus, pinv_cov = self._layer_stats[i]
                gaussian_score = self._mahalanobis_score_layer(
                    out_features, mus, pinv_cov
                )
                max_score = self.op.max(gaussian_score, dim=1)
                per_layer_scores.append(-self.op.convert_to_numpy(max_score))
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
            max_score = self.op.max(gaussian_score, dim=1)
            scores.append(-self.op.convert_to_numpy(max_score))

        if num_feature_layers > 1:
            aggregated_scores = self.aggregator.aggregate(scores)  # type: ignore
        else:
            aggregated_scores = scores[0]

        return aggregated_scores

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
