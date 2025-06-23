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

import numpy as np

from ..aggregator import BaseAggregator
from ..types import TensorType
from .base import FeatureBasedDetector


class SHE(FeatureBasedDetector):
    """
    "Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization
    with Modern Hopfield Energy"
    [link](https://openreview.net/forum?id=KkazG4lgKL)

    This method first computes the mean of the internal layer representation of ID data
    for each ID class. This mean is seen as the average of the ID activation patterns
    as defined in the original paper.
    The method then returns the maximum value of the dot product between the internal
    layer representation of the input and the average patterns, which is a simplified
    version of Hopfield energy as defined in the original paper. The per-layer
    confidence values can be combined through an aggregator to yield a single score

    Remarks:
    *   An input perturbation is applied in the same way as in ODIN score
    *   The original paper only considers the penultimate layer of the neural
    network, while we aggregate the results of multiple layers following different
    normalization strategies (see `BaseAggregator` for more details).

    Args:
        eps (float): Perturbation noise. Defaults to 0.0014.
        temperature (float, optional): Temperature parameter. Defaults to 1000.
        aggregator: Optional object implementing the `BaseAggregator` interface. It is
            used to combine the negative per-layer SHE scores returned by
            `_score_layer`.  If *None* and more than one layer is employed, a
            `StdNormalizedAggregator` is instantiated automatically.
    """

    def __init__(
        self,
        eps: float = 0.0014,
        temperature: float = 1000,
        aggregator: Optional[BaseAggregator] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            eps=eps, temperature=temperature, aggregator=aggregator, **kwargs
        )
        self.eps = eps
        self.temperature = temperature
        self.postproc_fns = None  # Will be set in `_fit_to_dataset`.

        # Fitted attributes
        self._classes: Optional[np.ndarray] = None
        self._layer_mus: List[TensorType] = []  # Shape per layer: [D, n_classes]

    # === Per-layer logic ===
    def _fit_layer(
        self,
        layer_id: int,
        layer_features: np.ndarray,
        info: dict,
        subset_size: int = 5000,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Compute mean vectors for a single layer and initial SHE scores.

        Args:
            layer_features: Tensor of shape `(N, D)` containing the flattened
                activations of in-distribution samples for one layer.
            labels_np: Ground-truth class labels as a `np.ndarray`.
            preds_np: Model predictions for the in-distribution samples.  Means
                are only computed from samples that are *both* predicted and
                labelled as the target class (following the original paper).
            subset_size: Number of samples on which to compute provisional OOD
                scores for the aggregator fit.  Ignored if *self.aggregator* is
                *None*.

        Returns:
            A tuple `(mus_layer, scores_layer)` where
            `mus_layer` is a tensor of shape `[D, n_classes]` storing the
            per-class mean vectors for the current layer and `scores_layer` is
            either *None* or a NumPy array of length *subset_size* containing
            the **negative** SHE scores on the first *subset_size* samples.
        """
        labels_np = info["labels"]
        preds_np = np.argmax(info["logits"], axis=1)

        if self._classes is None:
            self._classes = np.sort(np.unique(labels_np))

        mus_per_cls = []
        for cls in self._classes:
            idx = np.equal(labels_np, cls) & np.equal(preds_np, cls)
            feats_cls = layer_features[idx]
            mu = np.expand_dims(np.mean(feats_cls, axis=0), axis=0)
            mus_per_cls.append(mu)
        mus_layer = self.op.from_numpy(np.concatenate(mus_per_cls, axis=0))
        mus_layer = self.op.permute(mus_layer, (1, 0))

        self._layer_mus.append(mus_layer)

        scores_layer: Optional[np.ndarray] = None
        if getattr(self, "aggregator", None) is not None:
            n_samples = min(subset_size, layer_features.shape[0])
            feats_subset = self.op.from_numpy(layer_features[:n_samples])
            she_subset = self._score_layer(
                layer_id, feats_subset, {"logits": info["logits"]}
            )
            scores_layer = she_subset

        return scores_layer

    def _score_layer(
        self,
        layer_id: int,
        features: TensorType,
        info: dict,
        **kwargs,
    ) -> np.ndarray:
        """Compute *unnormalised* SHE confidence for a single layer."""

        mus_layer = self._layer_mus[layer_id]
        she = self.op.matmul(self.op.squeeze(features), mus_layer) / features.shape[1]
        she = self.op.max(she, dim=1)
        return -self.op.convert_to_numpy(she)

    # === Internal utilities ===

    # === Properties ===
    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.

        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        return True

    @property
    def requires_internal_features(self) -> bool:
        """
        Whether an OOD detector acts on internal model features.

        Returns:
            bool: True if the detector perform computations on an intermediate layer
            else False.
        """
        return True
