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
        orders: List[int] = [i for i in range(1, 6)],
    ):
        super().__init__(output_layers_id=output_layers_id)

        if isinstance(orders, int):
            orders = [orders]
        self.orders = orders

        self.postproc_fns_min_maxs = [
            self._min_maxs for i in range(len(output_layers_id))
        ]
        self.postproc_fns_stat = [self._stat for i in range(len(output_layers_id))]

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.

        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        return True

    def _fit_to_dataset(
        self,
        fit_dataset: Union[TensorType, DatasetType],
    ) -> None:
        """
        Constructs the index from ID data "fit_dataset", which will be used for
        nearest neighbor search.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        fit_feature_maps = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns_min_maxs
        )

        for i, feature_map in enumerate(fit_feature_maps):
            mins = self.op.unsqueeze(self.op.min(feature_map[:, 0, ...], dim=0), -1)
            maxs = self.op.unsqueeze(self.op.max(feature_map[:, 1, ...], dim=0), -1)
            min_max = self.op.cat([mins, maxs], dim=-1)
            fit_feature_maps[i] = min_max

        self.min_maxs = fit_feature_maps

        # In the paper, they use a separate validation data to compute
        # a normalization constant
        val_stats = self.feature_extractor.predict(
            fit_dataset,
            postproc_fns=self.postproc_fns_stat,
        )
        devnorm = self._deviation(val_stats)
        # For now, since class wise score is not available
        self.devnorm = [1.0 for i in range(len(self.min_maxs))]
        # self.devnorm = [float(self.op.mean(dev)) for dev in devnorm]
        # MAYBE PUT STAT AS PROCESS FCTS

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        tensor_stats = self.feature_extractor.predict(
            inputs, postproc_fns=self.postproc_fns_stat
        )
        tensor_dev = self._deviation(tensor_stats)
        score = self.op.mean(
            self.op.cat(
                [
                    self.op.unsqueeze(tensor_dev_l, dim=0) / devnorm_l
                    for tensor_dev_l, devnorm_l in zip(tensor_dev, self.devnorm)
                ]
            ),
            dim=0,
        )
        return np.array(score)

    def _deviation(self, feature_maps):
        deviation = []
        for feature_map, min_max in zip(feature_maps, self.min_maxs):
            where_min = self.op.where(feature_map < min_max[..., 0], 1.0, 0.0)
            where_max = self.op.where(feature_map > min_max[..., 1], 1.0, 0.0)
            deviation_min = (
                (min_max[..., 0] - feature_map)
                / self.op.abs(min_max[..., 0])
                * where_min
            )
            deviation_max = (
                (feature_map - min_max[..., 1])
                / self.op.abs(min_max[..., 1])
                * where_max
            )
            deviation.append(self.op.sum(deviation_min + deviation_max, dim=(1, 2)))
        return deviation

    def _stat(self, feature_map):
        fm_s = feature_map.shape
        stat = []
        for p in self.orders:
            feature_map_p = feature_map**p
            # construct the Gram matrix
            if len(fm_s) == 2:
                # prevents from computing a scalar per batch for layers of shape (1,)
                feature_map_p = self.op.einsum("bi,bj->b", feature_map_p, feature_map_p)
                feature_map_p = self.op.unsqueeze(feature_map_p, -1)
                feature_map_p = self.op.unsqueeze(feature_map_p, -1)
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
                    "bik,bjk->bij", feature_map_p, feature_map_p
                )
            feature_map_p = feature_map_p ** (1 / p)
            # get the lower triangular part of the matrix
            feature_map_p = self.op.tril(feature_map_p)
            # directly sum row-wise (to limit computational burden)
            feature_map_p = self.op.sum(feature_map_p, dim=-2)
            # stat.append(self.op.transpose(feature_map_p))
            stat.append(feature_map_p)
        stat = self.op.stack(stat, 1)
        return stat

    def _min_maxs(self, feature_map):
        stats = self._stat(feature_map)
        min_max_per_row = self.op.cat(
            [
                self.op.min(stats, dim=0, keepdim=True),
                self.op.max(stats, dim=0, keepdim=True),
            ],
            dim=0,
        )
        return self.op.unsqueeze(min_max_per_row, 0)
