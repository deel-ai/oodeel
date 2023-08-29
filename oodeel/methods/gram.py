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
import numpy as np

from ..types import DatasetType
from ..types import List
from ..types import TensorType
from ..types import Union
from .base import OODBaseDetector


class Gram(OODBaseDetector):
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
        output_layers_id: List[int] = [-2],
        orders: List[int] = [1],
    ):
        postproc_fns = [self.row_wise_sums for i in range(len(output_layers_id))]
        if isinstance(orders, int):
            orders = [orders]
        self.orders = orders
        super().__init__(output_layers_id=output_layers_id, postproc_fns=postproc_fns)

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.

        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        return True

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Constructs the index from ID data "fit_dataset", which will be used for
        nearest neighbor search.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        fit_feature_maps = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns
        )

        for i, feature_map in enumerate(fit_feature_maps):
            mins = self.op.unsqueeze(self.op.min(feature_map[..., 0], dim=0), -1)
            maxs = self.op.unsqueeze(self.op.max(feature_map[..., 1], dim=0), -1)
            min_max = self.op.cat([mins, maxs], dim=-1)
            fit_feature_maps[i] = min_max

        self.min_maxs = fit_feature_maps

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        input_projected = self.feature_extractor(inputs, postproc_fns=self.postproc_fns)

    def row_wise_sums(self, feature_map):
        fm_s = feature_map.shape

        min_max_per_order_row = []
        for p in self.orders:
            feature_map_p = feature_map**p
            # construct the Gram matrix
            if len(fm_s) == 2:
                # prevents from computing a scalar per batch for layers of shape (1,)
                feature_map_p = self.op.einsum(
                    "bi,bj->bij", feature_map_p, feature_map_p
                )
            elif len(fm_s) >= 3:
                # flatten the feature map
                if self.backend == "tensorflow":
                    feature_map_p = self.op.reshape(
                        feature_map_p, (fm_s[0], fm_s[-1], -1)
                    )
                else:
                    feature_map_p = self.op.reshape(
                        feature_map_p, (fm_s[0], fm_s[1], -1)
                    )
                feature_map_p = self.op.einsum(
                    "bij,bik->bjk", feature_map_p, feature_map_p
                )
            feature_map_p = feature_map_p ** (1 / p)
            # get the lower triangular part of the matrix
            feature_map_p = self.op.tril(feature_map_p, diagonal=-1)
            # directly sum row-wise (to limit computational burden)
            feature_map_p = self.op.sum(feature_map_p, dim=-2)
            min_max_per_row = self.op.cat(
                [
                    self.op.min(feature_map_p, dim=0, keepdim=True),
                    self.op.max(feature_map_p, dim=0, keepdim=True),
                ],
                dim=0,
            )
            min_max_per_order_row.append(self.op.transpose(min_max_per_row))
        return self.op.unsqueeze(self.op.stack(min_max_per_order_row), 0)
