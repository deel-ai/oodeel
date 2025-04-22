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
from typing import Union

import numpy as np
from sklearn.model_selection import train_test_split

from ..aggregator import BaseAggregator
from ..types import DatasetType
from ..types import TensorType
from .base import OODBaseDetector

EPSILON = 1e-6  # Numerical stability constant


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
        if isinstance(orders, int):
            orders = [orders]
        self.orders: List[int] = orders
        self.quantile = quantile
        self.aggregator = aggregator

        self.postproc_fns = None  # Will be set during fit
        # Mapping class -> list (per-layer) of thresholds [lower, upper]
        self.min_maxs: Dict[int, List[TensorType]] = {}
        # Normalisation constants when no aggregator is used
        self.devnorm: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stat(self, feature_map: TensorType) -> TensorType:
        """Compute Gram statistics for a single layer.

        Args:
            feature_map (TensorType): Feature map of shape `[B, ...]`.

        Returns:
            TensorType: Statistics of shape `[B, n_orders, C]`.
        """
        fm_shape = feature_map.shape
        stats = []
        for p in self.orders:
            # Raise the feature map to the specified order.
            fm_p = feature_map**p
            if len(fm_shape) == 2:
                # Dense layers: compute outer product.
                fm_p = self.op.einsum("bi,bj->bij", fm_p, fm_p)
            else:
                # Convolutional feature maps: flatten spatial dimensions.
                if self.backend == "tensorflow":
                    fm_p = self.op.reshape(
                        self.op.einsum("i...j->ij...", fm_p),
                        (fm_shape[0], fm_shape[-1], -1),
                    )
                else:
                    fm_p = self.op.reshape(fm_p, (fm_shape[0], fm_shape[1], -1))
                fm_p = self.op.matmul(fm_p, self.op.permute(fm_p, (0, 2, 1)))
            # Normalize and recover the original power.
            fm_p = self.op.sign(fm_p) * (self.op.abs(fm_p) ** (1 / p))
            # Use only the lower triangular part.
            fm_p = self.op.tril(fm_p)
            # Aggregate row-wise.
            fm_p = self.op.sum(fm_p, dim=2)
            stats.append(fm_p)
        return self.op.stack(stats, dim=1)

    def _deviation(self, stats: TensorType, thresholds: TensorType) -> TensorType:
        """Compute deviation of `stats` outside `thresholds`.

        Args:
            stats (TensorType): Gram stats, shape `[B, *, C]`.
            thresholds (TensorType): Lower & upper bounds, shape `[B, *, C, 2]`.

        Returns:
            TensorType: Deviation values, shape `[B]`.
        """
        below = self.op.where(stats < thresholds[..., 0], 1.0, 0.0)
        above = self.op.where(stats > thresholds[..., 1], 1.0, 0.0)
        dev_low = (
            (thresholds[..., 0] - stats) / (self.op.abs(thresholds[..., 0]) + EPSILON)
        ) * below
        dev_high = (
            (stats - thresholds[..., 1]) / (self.op.abs(thresholds[..., 1]) + EPSILON)
        ) * above
        return self.op.sum(dev_low + dev_high, dim=(1, 2))

    # ------------------------------------------------------------------
    # Per-layer helpers
    # ------------------------------------------------------------------

    def _fit_layer(
        self,
        layer_idx: int,
        train_stats: TensorType,
        val_stats: TensorType,
        train_preds: TensorType,
        val_preds: TensorType,
    ) -> Optional[np.ndarray]:
        """Fit thresholds for **one** layer and optionally return val scores.

        Args:
            layer_idx (int): Index of the processed layer.
            train_stats / val_stats (TensorType): Gram stats for train/val subsets.
            train_preds / val_preds (TensorType): Predicted labels.

        Returns:
            Optional[np.ndarray]: Validation deviations (if aggregator is used),
            otherwise `None`.
        """
        for cls in self._classes:
            idx = self.op.equal(train_preds, cls)
            lower = self.op.quantile(train_stats[idx], self.quantile, dim=0)
            upper = self.op.quantile(train_stats[idx], 1 - self.quantile, dim=0)
            self.min_maxs[cls][layer_idx] = self.op.cat(
                [self.op.unsqueeze(lower, -1), self.op.unsqueeze(upper, -1)], dim=-1
            )

        if self.aggregator is None:
            return None

        thr_batch = self.op.stack(
            [self.min_maxs[int(lbl)][layer_idx] for lbl in val_preds]
        )
        dev = self._deviation(val_stats, thr_batch)
        return self.op.convert_to_numpy(dev)

    def _score_layer(
        self, layer_idx: int, layer_stats: TensorType, preds: np.ndarray
    ) -> np.ndarray:
        """Score inputs for a **single** layer.

        Args:
            layer_idx (int): Layer index.
            layer_stats (TensorType): Gram stats.
            preds (np.ndarray): Predicted classes.

        Returns:
            np.ndarray: Deviation-based OOD scores.
        """
        thr_batch = self.op.stack([self.min_maxs[int(lbl)][layer_idx] for lbl in preds])
        dev = self._deviation(layer_stats, thr_batch)
        score = self.op.convert_to_numpy(dev)
        if self.aggregator is None and self.devnorm is not None:
            score = score / self.devnorm[layer_idx]
        return score

    # ------------------------------------------------------------------
    # Fit / score
    # ------------------------------------------------------------------

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
        n_layers = len(self.feature_extractor.feature_layers_id)
        self.postproc_fns = [self._stat] * n_layers

        stats_all, info = self.feature_extractor.predict(
            fit_dataset,
            postproc_fns=self.postproc_fns,
            return_labels=True,
            verbose=verbose,
        )
        preds_all = self.op.argmax(info["logits"], dim=1)
        self._classes = np.sort(np.unique(self.op.convert_to_numpy(preds_all))).tolist()
        self.min_maxs = {cls: [None] * n_layers for cls in self._classes}

        idx_all = np.arange(preds_all.shape[0])
        train_idx, val_idx = (
            train_test_split(idx_all, test_size=val_split, random_state=42)
            if val_split is not None
            else (idx_all, idx_all)
        )
        train_mask = self.op.from_numpy(np.isin(idx_all, train_idx))
        val_mask = self.op.from_numpy(np.isin(idx_all, val_idx))

        val_scores_for_agg = []
        for i in range(n_layers):
            val_scores = self._fit_layer(
                i,
                stats_all[i][train_mask],
                stats_all[i][val_mask],
                preds_all[train_mask],
                preds_all[val_mask],
            )
            if val_scores is not None:
                val_scores_for_agg.append(val_scores)

        if self.aggregator is not None:
            if val_scores_for_agg:
                self.aggregator.fit(val_scores_for_agg)
        else:
            devnorm = []
            for i in range(n_layers):
                per_cls = []
                for cls in self._classes:
                    cls_mask = self.op.equal(preds_all[val_mask], cls)
                    if self.op.sum(cls_mask) == 0:
                        continue
                    stats_cls = stats_all[i][val_mask][cls_mask]
                    thr = self.min_maxs[cls][i]
                    per_cls.append(float(self.op.mean(self._deviation(stats_cls, thr))))
                devnorm.append(np.mean(per_cls))
            self.devnorm = np.asarray(devnorm)

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
        layer_stats, logits = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )
        preds = self.op.convert_to_numpy(self.op.argmax(logits, dim=1))

        per_layer = [
            self._score_layer(i, layer_stats[i], preds) for i in range(len(layer_stats))
        ]

        if len(per_layer) > 1 and self.aggregator is not None:
            return self.aggregator.aggregate(per_layer)
        return np.mean(np.stack(per_layer, axis=1), axis=1)

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
