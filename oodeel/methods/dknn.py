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
from ..types import List
from ..types import TensorType
from ..types import Union
from .base import OODModel


class DKNN(OODModel):
    """
    "Out-of-Distribution Detection with Deep Nearest Neighbors"
    https://arxiv.org/abs/2204.06507

    Args:
        nearest: number of nearest neighbors to consider.
            Defaults to 1.
        output_layers_id: feature space on which to compute nearest neighbors.
            Defaults to [-2].
    """

    def __init__(
        self,
        nearest: int = 1,
        output_layers_id: List[int] = [-2],
    ):
        super().__init__(
            output_layers_id=output_layers_id,
        )

        self.index = None
        self.nearest = nearest

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]):
        """
        Constructs the index from ID data "fit_dataset", which will be used for
        nearest neighbor search.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        fit_projected = self.feature_extractor.predict(fit_dataset)
        fit_projected = self.op.convert_to_numpy(fit_projected)
        fit_projected = fit_projected.reshape(fit_projected.shape[0], -1)
        norm_fit_projected = self._l2_normalization(fit_projected)
        self.index = faiss.IndexFlatL2(norm_fit_projected.shape[1])
        self.index.add(norm_fit_projected)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        input_projected = self.feature_extractor(inputs)
        input_projected = self.op.convert_to_numpy(input_projected)
        input_projected = input_projected.reshape(input_projected.shape[0], -1)
        norm_input_projected = self._l2_normalization(input_projected)
        scores, _ = self.index.search(norm_input_projected, self.nearest)
        return scores[:, -1]

    def _l2_normalization(self, feat: np.ndarray) -> np.ndarray:
        return feat / (np.linalg.norm(feat, ord=2, axis=-1, keepdims=True) + 1e-10)
