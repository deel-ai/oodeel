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
from ..aggregator import StdNormalizedAggregator
from ..types import DatasetType
from ..types import TensorType
from .base import OODBaseDetector


class VIM(OODBaseDetector):
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
    ):
        super().__init__()
        # Store PCA settings and optional aggregator
        self._princ_dims = princ_dims
        self.pca_origin = pca_origin
        self.aggregator = aggregator
        # Containers for per-layer PCA parameters
        self.centers: List[TensorType] = []
        self.residual_projections: List[TensorType] = []
        self.eig_vals_list: List[np.ndarray] = []
        self.princ_dims_list: List[int] = []
        self.alphas: List[float] = []

    # === Public API (fit, score) ===
    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Fit PCA subspaces and compute scaling factors per feature layer.

        Steps per layer:
          1. Extract features, flatten to shape [N, D].
          2. Determine center: pseudo-center for final layer if requested or empirical
            mean.
          3. Compute empirical covariance and its eigen-decomposition.
          4. Select number of principal components (int or ratio).
          5. Store residual projector (eigenvectors for discarded dimensions).
          6. Compute alpha so that average residual norm matches average max-logit.
        """
        # Ensure post-processing functions extract features from each layer
        num_layers = len(self.feature_extractor.feature_layers_id)
        if self.postproc_fns is None:
            self.postproc_fns = [
                self.feature_extractor._default_postproc_fn for _ in range(num_layers)
            ]

        # Extract features and logits for all layers
        all_features, info = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, numpy_concat=True
        )
        logits_train = info["logits"]

        # Precompute max-logit energy baseline
        train_maxlogit = np.max(logits_train, axis=-1)

        # Fit PCA for each layer
        for idx, layer_id in enumerate(self.feature_extractor.feature_layers_id):
            # Flatten features: shape [N, D]
            feat = self.op.flatten(self.op.from_numpy(all_features[idx]))
            N, D = feat.shape

            # 1) Determine the subspace origin (center)
            if layer_id == -1 and self.pca_origin == "pseudo":
                # Use model's final linear layer weights to compute pseudo-center W^-1 b
                W, b = self.feature_extractor.get_weights(-1)
                W_mat = (
                    self.op.t(self.op.from_numpy(W))
                    if self.backend == "tensorflow"
                    else self.op.from_numpy(W)
                )
                b_vec = self.op.from_numpy(b.reshape(-1, 1))
                center = -self.op.reshape(
                    self.op.matmul(self.op.pinv(W_mat), b_vec), (-1,)
                )
            else:
                # Empirical mean for all other layers
                center = self.op.mean(feat, dim=0)

            # 2) Compute empirical covariance and eigen-decomposition
            centered = feat - center
            cov = self.op.matmul(self.op.t(centered), centered) / N
            eig_vals, eig_vecs = self.op.eigh(cov)
            eig_vals_np = self.op.convert_to_numpy(eig_vals)

            # 3) Select number of principal components
            if isinstance(self._princ_dims, int):
                assert (
                    0 < self._princ_dims < D
                ), f"princ_dims ({self._princ_dims}) must be in 1..{D-1}"
                princ_dim = self._princ_dims
            else:
                # Float: ratio of variance to retain
                assert (
                    0 < self._princ_dims <= 1
                ), f"princ_dims ratio ({self._princ_dims}) must be in (0,1]"
                # take princ_dim as the number of components that explain
                # self._princ_dims of the variance
                explained_variance = np.cumsum(np.flip(eig_vals_np)) / np.sum(
                    eig_vals_np
                )
                princ_dim = np.where(explained_variance > self._princ_dims)[0][0]
            # residual dimension = discarded components
            res_dim = D - princ_dim

            # Store PCA parameters
            self.centers.append(center)
            # residual projector: eigenvectors of smallest res_dim values
            self.residual_projections.append(eig_vecs[:, :res_dim])
            self.eig_vals_list.append(eig_vals_np)
            self.princ_dims_list.append(princ_dim)

            # 4) Compute scaling factor alpha for this layer
            residual_norms = self._compute_residual_score_tensor(feat, idx)
            alpha = float(np.mean(train_maxlogit) / np.mean(residual_norms))
            self.alphas.append(alpha)

        # 5) Aggregator setup for multi-layer combination
        if self.aggregator is None and num_layers > 1:
            self.aggregator = StdNormalizedAggregator()

        if self.aggregator is not None and num_layers > 1:
            # Gather per-layer OOD scores on training data to fit aggregator
            per_layer_scores = []
            for idx in range(num_layers):
                norms = self._compute_residual_score_tensor(
                    self.op.flatten(self.op.from_numpy(all_features[idx])), idx
                )
                ood_layer = self.alphas[idx] * norms - train_maxlogit
                per_layer_scores.append(ood_layer)
            self.aggregator.fit(per_layer_scores)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Compute the final VIM OOD score for input samples.

        Steps:
          1. Extract per-layer features and logits.
          2. Compute energy: log-sum-exp over logits.
          3. Compute residual norms per layer and combine:
             score_layer = alpha * residual_norm - energy.
          4. If multiple layers, aggregate via self.aggregator.

        Returns:
            Array of OOD scores (higher means more likely OOD).
        """
        # Extract features for each layer and the logits
        feats, logits = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )
        # Energy score: log-sum-exp of classifier logits
        energy = logsumexp(self.op.convert_to_numpy(logits), axis=-1)

        # Compute layer-wise scores
        scores: List[np.ndarray] = []
        for idx, f in enumerate(feats):
            flat = self.op.flatten(f)
            resid = self._compute_residual_score_tensor(flat, idx)
            score_layer = self.alphas[idx] * resid - energy
            scores.append(score_layer)

        # Aggregate if needed
        if len(scores) > 1 and self.aggregator is not None:
            return self.aggregator.aggregate(scores)  # type: ignore
        return scores[0]

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
        # Project onto residual subspace and compute vector norms
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
