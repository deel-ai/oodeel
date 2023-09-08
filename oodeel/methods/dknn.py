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

from ..types import DatasetType
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
            Defaults to 1.
    """

    def __init__(
        self,
        nearest: int = 1,
    ):
        super().__init__()

        self.index = None
        self.nearest = nearest

    def postproc_feature_maps(self, feature_map):
        if len(feature_map.shape) > 2:
            feature_map = self.op.avg_pool_2d(feature_map)
        return self.op.flatten(feature_map)

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Constructs the index from ID data "fit_dataset", which will be used for
        nearest neighbor search.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        self.postproc_fns = [
            self.postproc_feature_maps
            for i in range(len(self.feature_extractor.feature_layers_id))
        ]

        features, infos = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns
        )
        labels = infos["labels"]

        # unique sorted classes
        self._classes = np.sort(np.unique(self.op.convert_to_numpy(labels)))

        self._nn_indexes = []
        if not isinstance(features, list):
            features = [features]
        for feature in features:
            nn_indexes = dict()
            for cls in self._classes:
                indexes = self.op.equal(labels, cls)
                _feature_cls = feature[indexes]
                _feature_cls = self.op.convert_to_numpy(_feature_cls)
                _feature_cls = _feature_cls.reshape(_feature_cls.shape[0], -1)
                norm_feature = self._l2_normalization(_feature_cls)
                nn_indexes[cls] = faiss.IndexFlatL2(norm_feature.shape[1])
                nn_indexes[cls].add(norm_feature)
            self._nn_indexes.append(nn_indexes)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            Tuple[np.ndarray]: scores, logits
        """

        features, logits = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )
        preds = self.op.argmax(logits, dim=1)
        preds = self.op.convert_to_numpy(preds)

        scores_l = np.empty((0, preds.shape[0]))
        if not isinstance(features, list):
            features = [features]
        for nn_indexes, feature in zip(self._nn_indexes, features):
            feature_l = self.op.convert_to_numpy(feature)
            feature_l = feature_l.reshape(feature_l.shape[0], -1)
            norm_feature = self._l2_normalization(feature_l)
            scores = np.empty((0, preds.shape[0]))
            for cls in self._classes:
                scores_cls, _ = nn_indexes[cls].search(norm_feature, self.nearest)
                scores = np.concatenate([scores, np.expand_dims(scores_cls[:, -1], 0)])
            scores = scores[preds, np.arange(preds.shape[0])]
            scores_l = np.concatenate(
                [scores_l, np.expand_dims(scores, 0) / np.sqrt(feature_l.shape[1])]
            )
        return np.mean(scores_l, axis=0)

    def _l2_normalization(self, feat: np.ndarray) -> np.ndarray:
        """L2 normalization of a tensor along the last dimension.

        Args:
            feat (np.ndarray): the tensor to normalize

        Returns:
            np.ndarray: the normalized tensor
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
