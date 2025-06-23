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

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from ..aggregator import BaseAggregator
from ..types import TensorType
from .base import FeatureBasedDetector


class VIM(FeatureBasedDetector):
    """
    Virtual Matching Logit (VIM) out-of-distribution detector.

    Implements the VIM method from https://arxiv.org/abs/2203.10807:
      1. Energy-based score: log-sum-exp over classifier logits.
      2. PCA residual score: distance of features from a low-dimensional subspace.

    Supports multiple feature layers by computing PCA on each layer's features
    and combining per-layer VIM scores via an optional aggregator.

    Args:
        princ_dims (Union[int, float]):
            - If int: exact number of principal components to consider per layer.
            - If a float, it must be in [0,1), it represents the ratio of explained
                variance to consider to determine the number of principal components per
                layer. Defaults to 0.99.
        pca_origin (str): Method to compute the subspace origin (center).
            - "pseudo": (Only for the final layer (ID -1), other layers will use
                empirical mean!)
                Weights are used to compute the pseudo-center W⁻¹ b, where W is the
                weight matrix of the final linear layer (ID -1) and b is the bias
                vector.
            - "center": use the empirical mean of features. Defaults to "center".
        aggregator (Optional[BaseAggregator]): Combines multi-layer VIM scores.
            If None and more than one layer is used, defaults to
            StdNormalizedAggregator.
    """

    def __init__(
        self,
        princ_dims: Union[int, float] = 0.99,
        pca_origin: str = "center",
        aggregator: Optional[BaseAggregator] = None,
        **kwargs,
    ):
        super().__init__(aggregator=aggregator, **kwargs)
        # Store PCA settings and optional aggregator
        self._princ_dims = princ_dims
        self.pca_origin = pca_origin
        # Containers for per-layer PCA parameters
        self.centers: List[TensorType] = []
        self.residual_projections: List[TensorType] = []
        self.eig_vals_list: List[np.ndarray] = []
        self.princ_dims_list: List[int] = []
        self.alphas: List[float] = []

    # === Per-layer logic ===
    def _fit_layer(
        self,
        layer_id: int,
        layer_features: np.ndarray,
        info: dict,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Compute PCA statistics for a single feature layer.

        The PCA subspace is estimated from the layer activations of the
        in-distribution training data. The ratio between the norm of the
        residual component and the maximum logit defines the :math:`\alpha`
        scaling used at inference.

        Args:
            layer_id: Index of the feature layer.
            layer_features: Features for this layer with shape `[N, D]`.
            info: Dictionary containing at least the training logits.

        Returns:
            Optional[np.ndarray]: VIM scores of the training samples for this
            layer (used to fit an optional aggregator).
        """

        logits_train = info["logits"]
        train_maxlogit = np.max(logits_train, axis=-1)

        feat = self.op.flatten(self.op.from_numpy(layer_features))
        N, D = feat.shape

        if layer_id == -1 and self.pca_origin == "pseudo":
            W, b = self.feature_extractor.get_weights(-1)
            W_mat = (
                self.op.t(self.op.from_numpy(W))
                if self.backend == "tensorflow"
                else self.op.from_numpy(W)
            )
            b_vec = self.op.from_numpy(b.reshape(-1, 1))
            center = -self.op.reshape(self.op.matmul(self.op.pinv(W_mat), b_vec), (-1,))
        else:
            center = self.op.mean(feat, dim=0)

        centered = feat - center
        cov = self.op.matmul(self.op.t(centered), centered) / N
        eig_vals, eig_vecs = self.op.eigh(cov)
        eig_vals_np = self.op.convert_to_numpy(eig_vals)

        if isinstance(self._princ_dims, int):
            assert (
                0 < self._princ_dims < D
            ), f"princ_dims ({self._princ_dims}) must be in 1..{D-1}"
            princ_dim = self._princ_dims
        else:
            assert (
                0 < self._princ_dims <= 1
            ), f"princ_dims ratio ({self._princ_dims}) must be in (0,1]"
            explained_variance = np.cumsum(np.flip(eig_vals_np)) / np.sum(eig_vals_np)
            princ_dim = np.where(explained_variance > self._princ_dims)[0][0]

        proj = eig_vecs[:, : D - princ_dim]

        residual_norms = self._compute_residual_norms(feat, center, proj)
        alpha = float(np.mean(train_maxlogit) / np.mean(residual_norms))
        scores = alpha * residual_norms - train_maxlogit

        self.centers.append(center)
        self.residual_projections.append(proj)
        self.eig_vals_list.append(eig_vals_np)
        self.princ_dims_list.append(princ_dim)
        self.alphas.append(alpha)

        return scores

    def _score_layer(
        self,
        layer_id: int,
        layer_features: TensorType,
        info: dict,
        **kwargs,
    ) -> np.ndarray:
        """Compute the VIM score associated with one feature layer.

        Args:
            layer_id: Index of the processed layer.
            layer_features: Features from the current layer.
            info: Dictionary containing the logits of the batch.

        Returns:
            np.ndarray: VIM scores for the layer.
        """
        energy = logsumexp(self.op.convert_to_numpy(info["logits"]), axis=-1)
        flat = self.op.flatten(layer_features)
        resid = self._compute_residual_score_tensor(flat, layer_id)
        return self.alphas[layer_id] * resid - energy

    # === Internal utilities ===
    def _compute_residual_score_tensor(
        self, features: TensorType, layer_idx: int
    ) -> np.ndarray:
        """
        Compute the residual norm of features orthogonal to the principal subspace.

        Args:
            features: Flattened feature matrix [N, D].
            layer_idx: Index of the feature layer.
        Returns:
            Numpy array of residual norms (shape [N]).
        """
        center = self.centers[layer_idx]
        proj = self.residual_projections[layer_idx]
        return self._compute_residual_norms(features, center, proj)

    def _compute_residual_norms(
        self, features: TensorType, center: TensorType, proj: TensorType
    ) -> np.ndarray:
        """Compute residual norms for the provided features.

        Args:
            features: Flattened feature matrix `[N, D]`.
            center: Center of the PCA subspace for the layer.
            proj: Projection matrix onto the residual subspace.

        Returns:
            Numpy array of residual norms of shape `[N]`.
        """
        coords = self.op.matmul(features - center, proj)
        norms = self.op.norm(coords, dim=-1)
        return self.op.convert_to_numpy(norms)

    # === Visualization ===
    def plot_spectrum(self) -> None:
        """
        Visualize residual explained variance per layer vs. principal dimensions being
        excluded.

        If princ_dims is int: x-axis = number of components [0..D-1].
        If princ_dims is float: x-axis = ratio [0..1].

        Draws:
          - Curve: residual explained variance vs. number of principal components.
          - Dashed line: selected princ_dims marker.
        """
        is_ratio = isinstance(self._princ_dims, float)

        for idx, eig_vals in enumerate(self.eig_vals_list):
            D = eig_vals.size
            # Compute residual explained variance curve
            residual_cumsum = np.cumsum(eig_vals)[::-1]
            residual_explained = residual_cumsum / residual_cumsum.max()

            # Choose x-axis scale and marker
            if is_ratio:
                x = np.linspace(0, 1, D)
                marker = self.princ_dims_list[idx] / D
                xlabel = "Ratio of principal components"
            else:
                x = np.arange(D)
                marker = self.princ_dims_list[idx]
                xlabel = "Number of principal components"

            (line,) = plt.plot(x, residual_explained, label=f"layer {idx}")
            plt.axvline(
                x=marker,
                linestyle="--",
                color=line.get_color(),
                label=f"layer {idx} marker",
            )

        plt.xlabel(xlabel)
        plt.ylabel("Residual explained variance")
        plt.legend()
        plt.tight_layout()
        plt.show()

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
