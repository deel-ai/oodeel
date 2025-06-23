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
from .base import FeatureBasedDetector

EPSILON = 1e-6  # Numerical stability constant


class Gram(FeatureBasedDetector):
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
            If multiple layers are used and no aggregator is provided,
            StdNormalizedAggregator is used by default. Defaults to None.
    """

    def __init__(
        self,
        orders: Union[List[int], int] = list(range(1, 6)),
        quantile: float = 0.01,
        aggregator: Optional[BaseAggregator] = None,
        **kwargs,
    ):
        super().__init__(aggregator=aggregator, **kwargs)
        if isinstance(orders, int):
            orders = [orders]
        self.orders: List[int] = orders
        self.quantile = quantile

        self.postproc_fns = None  # Will be set during fit
        # Mapping class -> list (per-layer) of thresholds [lower, upper]
        self.min_maxs: Dict[int, List[TensorType]] = {}

    # === Public API (override of _fit_to_dataset) ===
    def _fit_to_dataset(
        self,
        fit_dataset: DatasetType,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Fit thresholds on Gram statistics from a dataset.

        This method sets :attr:`postproc_fns` to compute Gram matrices for all
        selected feature layers and then delegates the actual fitting to the
        generic implementation in :class:`OODBaseDetector`.

        Args:
            fit_dataset: Dataset containing in-distribution samples.
            verbose: Whether to display a progress bar during feature extraction.
            **kwargs: Additional keyword arguments forwarded to :func:`_fit_layer`.

        Returns:
            None
        """
        n_layers = len(self.feature_extractor.feature_layers_id)
        if self.postproc_fns is None:
            self.postproc_fns = [self._stat] * n_layers

        super()._fit_to_dataset(fit_dataset, verbose=verbose, **kwargs)

    # === Per-layer logic ===
    def _fit_layer(
        self,
        layer_id: int,
        layer_stats: np.ndarray,
        info: dict,
        val_split: float = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Fit thresholds for one layer and optionally return validation scores.

        Args:
            layer_id: Index of the processed layer.
            layer_stats: Gram statistics for this layer.
            info: Dictionary containing the logits of the training data.
            val_split: Ratio of samples used for aggregator fitting.

        Returns:
            Optional[np.ndarray]: Validation deviations if an aggregator is used.
        """

        preds_all = np.argmax(info["logits"], axis=1)

        # initialize min_maxs if not already done.
        if not self.min_maxs:
            n_layers = len(self.feature_extractor.feature_layers_id)
            self._classes = np.sort(np.unique(preds_all)).tolist()
            self.min_maxs = {cls: [None] * n_layers for cls in self._classes}

        # split the dataset into training and validation sets (as in original paper).
        idx_all = np.arange(preds_all.shape[0])
        train_idx, val_idx = (
            train_test_split(idx_all, test_size=val_split, random_state=42)
            if val_split is not None
            else (idx_all, idx_all)
        )

        train_stats = layer_stats[train_idx]
        val_stats = layer_stats[val_idx]
        train_preds = preds_all[train_idx]
        val_preds = preds_all[val_idx]

        # compute min/max thresholds for each class
        for cls in self._classes:
            cls_mask = train_preds == cls
            stats_cls_np = train_stats[cls_mask]
            stats_cls_t = self.op.from_numpy(stats_cls_np)
            lower = self.op.quantile(stats_cls_t, self.quantile, dim=0)
            upper = self.op.quantile(stats_cls_t, 1 - self.quantile, dim=0)
            self.min_maxs[cls][layer_id] = self.op.cat(
                [self.op.unsqueeze(lower, -1), self.op.unsqueeze(upper, -1)],
                dim=-1,
            )

        if getattr(self, "aggregator", None) is None:
            return None

        batch_size = 128
        N = val_stats.shape[0]
        val_dev = []
        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_idx = np.arange(start, min(end, N))
            stats_t = self.op.from_numpy(val_stats[batch_idx])
            thr_list = [self.min_maxs[int(val_preds[i])][layer_id] for i in batch_idx]
            thr_t = self.op.stack(thr_list, dim=0)
            dev_t = self._deviation(stats_t, thr_t)
            val_dev.append(self.op.convert_to_numpy(dev_t))

        return np.concatenate(val_dev, axis=0)

    def _score_layer(
        self,
        layer_id: int,
        layer_stats: TensorType,
        info: dict,
        **kwargs,
    ) -> np.ndarray:
        """Score inputs for a single layer.

        Args:
            layer_id (int): Layer index.
            layer_stats (TensorType): Gram stats.
            info (dict): Dictionary containing auxiliary data, such as logits.
        Returns:
            np.ndarray: Deviation-based OOD scores.
        """
        preds = np.argmax(self.op.convert_to_numpy(info["logits"]), axis=1)
        thr_batch = self.op.stack([self.min_maxs[int(lbl)][layer_id] for lbl in preds])
        dev = self._deviation(layer_stats, thr_batch)
        return self.op.convert_to_numpy(dev)

    # === Internal utilities ===
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

    # === Properties ===
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
