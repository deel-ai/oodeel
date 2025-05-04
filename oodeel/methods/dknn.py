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
from ..aggregator import StdNormalizedAggregator
from ..types import DatasetType
from ..types import Optional
from ..types import TensorType
from ..types import Tuple
from ..types import Union
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

        if self.use_gpu:
            try:
                self.res = faiss.StandardGpuResources()
            except AttributeError as e:
                raise ImportError(
                    "faiss-gpu is not installed, but use_gpu was set to True."
                    + "Please install faiss-gpu or set use_gpu to False."
                ) from e

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

    def _fit_layer(
        self, layer_features: np.ndarray
    ) -> Tuple[faiss.IndexFlatL2, Optional[np.ndarray]]:
        """
        Build a FAISS index for a single feature layer and compute initial scores for
        aggregator fitting.

        Args:
            layer_features (TensorType): Feature tensor corresponding to a specific
                layer.

        Returns:
            Tuple[faiss.IndexFlatL2, Optional[np.ndarray]]:
                - The constructed FAISS index for the layer.
                - Scores computed on a subset of features (first 1000 samples) if an
                    aggregator is used; otherwise, None.
        """
        norm_features = self._prepare_layer_features(layer_features)
        index = self._create_index(norm_features.shape[1])
        index.add(norm_features)

        scores = None
        if self.aggregator is not None:
            # Use only a subset of samples for computing initial scores, ensuring at
            # least 2 neighbors (distance to self is 0)
            scores_subset, _ = index.search(norm_features[:1000], max(self.nearest, 2))
            scores = scores_subset[:, -1]
        return index, scores

    def _score_layer(
        self, index: faiss.IndexFlatL2, layer_features: TensorType
    ) -> np.ndarray:
        """
        Compute the OOD scores for a single feature layer using the provided FAISS
        index.

        Args:
            index (faiss.IndexFlatL2): Precomputed FAISS index for the feature layer.
            layer_features (TensorType): Feature tensor for the layer to be scored.

        Returns:
            np.ndarray: The OOD scores calculated as the distance to the k-th nearest
                neighbor.
        """
        layer_features = self.op.convert_to_numpy(layer_features)
        norm_features = self._prepare_layer_features(layer_features)
        scores, _ = index.search(norm_features, self.nearest)
        return scores[:, -1]

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Fit the detector on an in-distribution dataset by building FAISS indices for
        each feature layer.

        The method performs the following steps:
          1. Extract features from the input dataset using the feature extractor.
          2. For each feature layer:
             a. Prepare the feature data.
             b. Create and populate a FAISS index.
             c. Optionally compute scores for aggregator fitting.
          3. If an aggregator is used (or needed when multiple layers exist), fit it
            using the computed scores.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): In-distribution data used for
                constructing the indices.
        """
        self.indexes = []
        num_feature_layers = len(self.feature_extractor.feature_layers_id)

        if self.postproc_fns is None:
            self.postproc_fns = [
                self.feature_extractor._default_postproc_fn
            ] * num_feature_layers

        fit_projected, _ = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, numpy_concat=True
        )
        per_layer_scores = []

        # If there is more than one feature layer, ensure an aggregator is defined.
        if self.aggregator is None and num_feature_layers > 1:
            self.aggregator = StdNormalizedAggregator()

        for i in range(num_feature_layers):
            index, layer_scores = self._fit_layer(fit_projected[i])
            self.indexes.append(index)
            if self.aggregator is not None:
                per_layer_scores.append(layer_scores)

        if self.aggregator is not None:
            self.aggregator.fit(per_layer_scores)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Compute the OOD scores for the given inputs by aggregating scores from each
        feature layer.

        The scoring process includes:
          1. Extracting features for each layer from the input samples.
          2. Computing the distance to the k-th nearest neighbor using the precomputed
            FAISS index.
          3. Aggregating per-layer scores using the specified aggregator.

        Args:
            inputs (TensorType): Input samples to be scored.

        Returns:
            Tuple[np.ndarray]: A tuple containing the aggregated OOD scores.
        """
        input_projected, _ = self.feature_extractor.predict_tensor(
            inputs,
            postproc_fns=self.postproc_fns,
        )
        scores = []

        for i, index in enumerate(self.indexes):
            scores.append(self._score_layer(index, input_projected[i]))

        if self.aggregator is not None:
            aggregated_scores = self.aggregator.aggregate(scores)
        elif len(scores) == 1:
            aggregated_scores = scores[0]
        else:
            raise ValueError(
                "DKNN requires an aggregator to be defined when using multiple layers."
            )
        return aggregated_scores

    def _l2_normalization(self, feat: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to an array of feature vectors along the last dimension.

        Args:
            feat (np.ndarray): Input array of features.

        Returns:
            np.ndarray: L2-normalized feature array.
        """
        return feat / (np.linalg.norm(feat, ord=2, axis=-1, keepdims=True) + 1e-10)

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
