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
import faiss
import numpy as np

from ..aggregator import BaseAggregator
from ..types import Optional
from ..types import TensorType
from .base import OODBaseDetector


class DKNN(OODBaseDetector):
    """
    "Out-of-Distribution Detection with Deep Nearest Neighbors"
    https://arxiv.org/abs/2204.06507

    Args:
        nearest: number of nearest neighbors to consider.
            Defaults to 50.
        use_gpu (bool): Whether to enable GPU acceleration for FAISS. Defaults to False.
        aggregator (Optional[BaseAggregator]): Aggregator to combine scores from
            multiple feature layers. If not provided and multiple layers are used, a
            StdNormalizedAggregator will be employed.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(
        self,
        nearest: int = 50,
        use_gpu: bool = False,
        aggregator: BaseAggregator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nearest = nearest
        self.use_gpu = use_gpu
        self.aggregator = aggregator
        self.indexes: list[faiss.IndexFlatL2] = []

        if self.use_gpu:
            try:
                self.res = faiss.StandardGpuResources()
            except AttributeError as e:
                raise ImportError(
                    "faiss-gpu is not installed, but use_gpu was set to True."
                    + "Please install faiss-gpu or set use_gpu to False."
                ) from e

    # === Per-layer logic ===
    def _fit_layer(
        self,
        layer_id: int,
        layer_features: np.ndarray,
        info: dict,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Fit one FAISS index and optionally return aggregator scores.

        The extracted features are L2-normalized and stored into a FAISS index
        dedicated to the current layer. When an `aggregator` is used, distances
        to the :math:`k`-th nearest neighbour are computed on a small subset of
        samples and returned for later normalization.

        Args:
            layer_id: Index of the processed layer.
            layer_features: Feature tensor corresponding to that layer.
            info: Dictionary of auxiliary data (unused).

        Returns:
            Optional[np.ndarray]: Distances to the :math:`k`-th neighbour for the
                subset, or `None` if no aggregator is involved.
        """
        norm_features = self._prepare_layer_features(layer_features)
        index = self._create_index(norm_features.shape[1])
        index.add(norm_features)

        self.indexes.append(index)

        scores = None
        if getattr(self, "aggregator", None) is not None:
            # Compute KNN scores on a subset of samples (first 1000) to fit the
            # aggregator. We use k = max(self.nearest, 2) to avoid trivial zero
            # distances caused by querying each point against itself when k=1.
            # Using k >= 2 ensures the score reflects distance to a true neighbor.
            scores_subset, _ = index.search(norm_features[:1000], max(self.nearest, 2))
            scores = scores_subset[:, -1]
        return scores

    def _score_layer(
        self,
        layer_id: int,
        layer_features: TensorType,
        info: dict,
        **kwargs,
    ) -> np.ndarray:
        """Compute KNN scores for a single feature layer.

        Args:
            layer_id: Index of the processed layer.
            layer_features: Feature tensor associated with this layer.
            info: Dictionary of auxiliary data (unused).

        Returns:
            np.ndarray: Distance to the :math:`k`-th nearest neighbour.
        """
        index = self.indexes[layer_id]
        layer_features = self.op.convert_to_numpy(layer_features)
        norm_features = self._prepare_layer_features(layer_features)
        scores, _ = index.search(norm_features, self.nearest)
        return scores[:, -1]

    # === Internal utilities ===
    def _prepare_layer_features(self, features: np.ndarray) -> np.ndarray:
        """
        Convert a feature tensor to a 2D numpy array and apply L2 normalization.

        Args:
            features (np.ndarray): Feature tensor to be processed.

        Returns:
            np.ndarray: Processed feature array with shape (num_samples, feature_dim)
                and L2 normalized.
        """
        features = features.reshape(features.shape[0], -1)
        return self._l2_normalization(features)

    def _create_index(self, dim: int) -> faiss.IndexFlatL2:
        """
        Create a FAISS index for features of a given dimensionality.

        Args:
            dim (int): Dimensionality of the feature vectors.

        Returns:
            faiss.IndexFlatL2: A FAISS index instance, using GPU acceleration if
                enabled.
        """
        if self.use_gpu:
            cpu_index = faiss.IndexFlatL2(dim)
            return faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        else:
            return faiss.IndexFlatL2(dim)

    def _l2_normalization(self, feat: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to an array of feature vectors along the last dimension.

        Args:
            feat (np.ndarray): Input array of features.

        Returns:
            np.ndarray: L2-normalized feature array.
        """
        return feat / (np.linalg.norm(feat, ord=2, axis=-1, keepdims=True) + 1e-10)

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
