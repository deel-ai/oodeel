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
from typing import Union

import numpy as np
from sklearn.model_selection import train_test_split

from ..aggregator import BaseAggregator
from ..types import DatasetType
from ..types import TensorType
from .base import OODBaseDetector


EPSILON = 1e-6  # Constant for numerical stability


class Gram(OODBaseDetector):
    r"""
    "Detecting Out-of-Distribution Examples with Gram Matrices"
    [link](https://proceedings.mlr.press/v119/sastry20a.html)

    **Important Disclaimer**: Taking the statistics of min/max deviation, as in the
    paper, raises some problems. The method may yield a score of zero for some tasks
    because the sample extreme values become more extreme with larger sample sizes.
    To mitigate this, we replace the min/max with the q / (1-q) quantile threshold,
    where q is a parameter that controls the discriminative ability of the method.

    This approach improved baseline performance in our experiments.

    Args:
        orders (Union[List[int], int]): Power orders to consider for the correlation
            matrix. If an int is provided, it is converted to a list.
        quantile (float): Quantile to consider for the correlations to build the
            deviation threshold.
        aggregator (Optional[BaseAggregator]): Aggregator to combine multi-layer scores.
            If multiple layers are used and no aggregator is provided, the per-layer
            scores are aggregated by taking their mean (normalized by precomputed
            deviation scores on the validation set, as in the original paper).
            Defaults to None.
    """

    def __init__(
        self,
        orders: Union[List[int], int] = list(range(1, 6)),
        quantile: float = 0.01,
        aggregator: Optional[BaseAggregator] = None,
    ):
        super().__init__()
        # Ensure orders is a list even if a single int is provided.
        if isinstance(orders, int):
            orders = [orders]
        self.orders: List[int] = orders
        self.quantile = quantile
        self.aggregator = aggregator

        # postproc_fns is set during fitting to the Gram statistic function.
        self.postproc_fns = None

        # Dictionary mapping class -> list (per-layer) of quantile thresholds.
        self.min_maxs = {}

        # When no aggregator is provided, normalization constants for per-layer
        # deviations.
        self.devnorm = None

    def _fit_to_dataset(
        self,
        fit_dataset: Union[TensorType, DatasetType],
        val_split: float = 0.2,
        verbose: bool = False,
    ) -> None:
        """
        Fit the detector on in-distribution data by computing per-layer and per-class
        quantile thresholds for the Gram statistics. Depending on whether an aggregator
        is provided, either fit an aggregator on a validation subset or compute
        normalization constants per layer.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): In-distribution data.
            val_split (float): Fraction of data to use as validation for aggregator
                fitting (as in the original paper). If None, the entire fit dataset is
                used for both quantile threshold computation and aggregator fitting.
                Defaults to 0.2.
            verbose (bool): Whether to print additional information.
        """
        num_layers = len(self.feature_extractor.feature_layers_id)
        # Set the postprocessing functions to compute Gram statistics for each layer.
        self.postproc_fns = [self._stat for _ in range(num_layers)]

        # Compute Gram statistics and obtain prediction info.
        fit_stats, info = self.feature_extractor.predict(
            fit_dataset,
            postproc_fns=self.postproc_fns,
            return_labels=True,
            verbose=verbose,
        )
        preds = self.op.argmax(info["logits"], dim=1)
        self._classes = np.sort(np.unique(self.op.convert_to_numpy(preds))).tolist()

        full_indices = np.arange(preds.shape[0])

        # Split indices into training and validation sets.
        if val_split is not None:
            train_indices, val_indices = train_test_split(
                full_indices, test_size=val_split, random_state=42
            )
            # Create boolean masks for training and validation sets.
            train_mask = self.op.from_numpy(np.isin(full_indices, train_indices))
            val_mask = self.op.from_numpy(np.isin(full_indices, val_indices))

            train_stats = [fs[train_mask] for fs in fit_stats]
            val_stats = [fs[val_mask] for fs in fit_stats]
            train_preds = preds[train_mask]
            val_preds = preds[val_mask]
        else:
            train_preds = preds
            train_stats = fit_stats
            val_stats = fit_stats
            val_preds = preds

        # Compute quantile thresholds for each class and each layer.
        self.min_maxs = {}
        for cls in self._classes:
            indices = self.op.equal(train_preds, cls)
            cls_thresholds = []
            for layer_stats in train_stats:
                lower = self.op.quantile(layer_stats[indices], self.quantile, dim=0)
                upper = self.op.quantile(layer_stats[indices], 1 - self.quantile, dim=0)
                # Unsqueeze to maintain a consistent shape, then concatenate.
                lower = self.op.unsqueeze(lower, -1)
                upper = self.op.unsqueeze(upper, -1)
                cls_thresholds.append(self.op.cat([lower, upper], dim=-1))
            self.min_maxs[cls] = cls_thresholds

        # Either fit the aggregator or compute normalization constants.
        if self.aggregator is not None:
            per_layer_scores = []
            for layer_idx in range(num_layers):
                # Collect the thresholds corresponding to each sample's predicted class.
                thresholds = self.op.stack(
                    [self.min_maxs[pred.item()][layer_idx] for pred in val_preds]
                )
                deviation = self._deviation([val_stats[layer_idx]], [thresholds])[0]
                # Use negative deviation so that higher scores indicate OOD.
                per_layer_scores.append(self.op.convert_to_numpy(deviation))
            self.aggregator.fit(per_layer_scores)
        else:
            # if no aggregator is provided, compute normalization constants as in
            # the original paper.
            devnorm_list = []
            for cls in self._classes:
                cls_devnorm = []
                for layer_idx in range(num_layers):
                    # Select validation statistics corresponding to the current class.
                    val_stats_cls = [
                        val_stats[layer_idx][self.op.equal(val_preds, cls)]
                    ]
                    thresholds = [self.min_maxs[cls][layer_idx]]
                    dev = self._deviation(val_stats_cls, thresholds)[0]
                    cls_devnorm.append(float(self.op.mean(dev)))
                devnorm_list.append(cls_devnorm)
            self.devnorm = np.mean(np.array(devnorm_list), axis=0)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Compute OOD scores for input samples. For each layer, calculate the deviation
        from stored quantile thresholds. If an aggregator is available (and there is
        more than one layer), combine per-layer scores via the aggregator; otherwise,
        average the normalized scores.

        Args:
            inputs (TensorType): Input samples to be scored.

        Returns:
            np.ndarray: Final OOD scores.
        """
        tensor_stats, logits = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )
        preds = self.op.convert_to_numpy(self.op.argmax(logits, dim=1))

        per_layer_scores = []
        num_layers = len(tensor_stats)
        for i in range(num_layers):
            # Gather stored thresholds based on the predicted class for each sample.
            thresholds = self.op.stack([self.min_maxs[label][i] for label in preds])
            deviation = self._deviation([tensor_stats[i]], [thresholds])[0]
            score_layer = self.op.convert_to_numpy(deviation)
            # Normalize score if no aggregator was provided.
            if self.aggregator is None:
                score_layer = score_layer / self.devnorm[i]
            per_layer_scores.append(score_layer)
        if num_layers > 1 and self.aggregator is not None:
            aggregated_scores = self.aggregator.aggregate(per_layer_scores)
        else:
            aggregated_scores = np.mean(np.stack(per_layer_scores, axis=1), axis=1)
        return aggregated_scores

    def _deviation(
        self, stats: List[TensorType], min_maxs: List[TensorType]
    ) -> List[TensorType]:
        """
        Compute the normalized deviation for each layer with respect to the stored
        quantile thresholds. For each sample and each feature, deviation is computed if
        the statistic falls outside the [lower, upper] interval.

        Args:
            stats (List[TensorType]): List of Gram statistics (one per layer).
            min_maxs (List[TensorType]): List of corresponding min/max thresholds (one
                per layer).

        Returns:
            List[TensorType]: Per-sample deviation values for each layer.
        """
        deviations = []
        for stat, min_max in zip(stats, min_maxs):
            below_mask = self.op.where(stat < min_max[..., 0], 1.0, 0.0)
            above_mask = self.op.where(stat > min_max[..., 1], 1.0, 0.0)
            deviation_lower = (
                (min_max[..., 0] - stat) / (self.op.abs(min_max[..., 0]) + EPSILON)
            ) * below_mask
            deviation_upper = (
                (stat - min_max[..., 1]) / (self.op.abs(min_max[..., 1]) + EPSILON)
            ) * above_mask
            deviation = self.op.sum(deviation_lower + deviation_upper, dim=(1, 2))
            deviations.append(deviation)
        return deviations

    def _stat(self, feature_map: TensorType) -> TensorType:
        """
        Compute Gram matrix–based statistics for a given feature map. For each power
        order, the feature map is raised to that power, a Gram matrix is computed, and
        statistics are derived from the lower triangular portion.

        Args:
            feature_map (TensorType): Input feature map.

        Returns:
            TensorType: Stacked Gram statistics with shape [batch, n_orders, channel].
        """
        fm_shape = feature_map.shape
        stats = []
        for p in self.orders:
            # Raise the feature map to the specified order.
            feature_map_p = feature_map**p
            if len(fm_shape) == 2:
                # Dense layers: compute outer product.
                feature_map_p = self.op.einsum(
                    "bi,bj->bij", feature_map_p, feature_map_p
                )
            elif len(fm_shape) >= 3:
                # Convolutional feature maps: flatten spatial dimensions.
                if self.backend == "tensorflow":
                    feature_map_p = self.op.reshape(
                        self.op.einsum("i...j->ij...", feature_map_p),
                        (fm_shape[0], fm_shape[-1], -1),
                    )
                else:
                    feature_map_p = self.op.reshape(
                        feature_map_p, (fm_shape[0], fm_shape[1], -1)
                    )
                feature_map_p = self.op.matmul(
                    feature_map_p, self.op.permute(feature_map_p, (0, 2, 1))
                )
            # Normalize and recover the original power.
            feature_map_p = self.op.sign(feature_map_p) * (
                self.op.abs(feature_map_p) ** (1 / p)
            )
            # Use only the lower triangular part.
            feature_map_p = self.op.tril(feature_map_p)
            # Aggregate row-wise.
            feature_map_p = self.op.sum(feature_map_p, dim=2)
            stats.append(feature_map_p)
        return self.op.stack(stats, 1)

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True
