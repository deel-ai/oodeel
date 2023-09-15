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
from sklearn.model_selection import train_test_split

from ..types import DatasetType
from ..types import List
from ..types import TensorType
from ..types import Union
from .base import OODBaseDetector


class Gram(OODBaseDetector):
    r"""
    "Detecting Out-of-Distribution Examples with Gram Matrices"
    [link](https://proceedings.mlr.press/v119/sastry20a.html)

    **Important Disclaimer**: Taking the statistics of min/max deviation,
    as in the paper raises some problems.

    The method often yields a score of zero for some tasks.
    This is expected since the min/max among the samples of a random
    variable becomes more and more extreme with the sample
    size. As a result, computing the min/max over the training set is likely to produce
    min/max values that are so extreme that none of the in distribution correlations of
    the validation set goes beyond these threshold. The worst is that a significant
    part of ood data does not exceed the thresholds either. This can be aleviated by
    computing the min/max over a limited number of sample. However, it is
    counter-intuitive and, in our opinion, not desirable: adding
    some more information should only improve a method.

    Hence, we decided to replace the min/max by the q / 1-q quantile, with q a new
    parameter of the method. Specifically, instead of the deviation as defined in
    eq. 3 of the paper, we use the definition
    $$
    \delta(t_q, t_{1-q}, value) =
    \begin{cases}
        0 & \text{if} \; t_q \leq value \leq t_{1-q},  \;\;
        \frac{t_q - value}{|t_q|} & \text{if } value < t_q,  \;\;
        \frac{value - t_{1-q}}{|t_q|} & \text{if } value > t_{1-q}
    \end{cases}
    $$
    With this new deviation, the more point we add, the more accurate the quantile
    becomes. In addition, the method can be made more or less discriminative by
    toggling the value of q.

    Finally, we found that this approach improved the performance of the baseline in
    our experiments.

    Args:
        orders (List[int]): power orders to consider for the correlation matrix
        quantile (float): quantile to consider for the correlations to build the
            deviation threshold.

    """

    def __init__(
        self,
        orders: List[int] = [i for i in range(1, 11)],
        quantile: float = 0.01,
    ):
        super().__init__()
        if isinstance(orders, int):
            orders = [orders]
        self.orders = orders
        self.postproc_fns = None
        self.quantile = quantile

    def _fit_to_dataset(
        self,
        fit_dataset: Union[TensorType, DatasetType],
        val_split: float = 0.2,
    ) -> None:
        """
        Compute the quantiles of channelwise correlations for each layer, power of
        gram matrices, and class. Then, compute the normalization constants for the
        deviation. To stay faithful to the spirit of the original method, we still name
        the quantiles min/max

        Args:
            fit_dataset (Union[TensorType, DatasetType]): input dataset (ID) to
                construct the index with.
            val_split (float): The percentage of fit data to use as validation data for
                normalization. Default to 0.2.
        """
        self.postproc_fns = [
            self._stat for i in range(len(self.feature_extractor.feature_layers_id))
        ]

        fit_stats, info = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns, return_labels=True
        )
        labels = info["labels"]
        self._classes = np.sort(np.unique(self.op.convert_to_numpy(labels)))

        full_indices = np.arange(labels.shape[0])
        train_indices, val_indices = train_test_split(full_indices, test_size=val_split)
        train_indices = self.op.from_numpy(
            [bool(ind in train_indices) for ind in full_indices]
        )
        val_indices = self.op.from_numpy(
            [bool(ind in val_indices) for ind in full_indices]
        )

        val_stats = [fit_stat[val_indices] for fit_stat in fit_stats]
        fit_stats = [fit_stat[train_indices] for fit_stat in fit_stats]
        labels = labels[train_indices]

        self.min_maxs = dict()
        for cls in self._classes:
            indexes = self.op.equal(labels, cls)
            min_maxs = []
            for fit_stat in fit_stats:
                fit_stat = fit_stat[indexes]
                mins = self.op.unsqueeze(
                    self.op.quantile(fit_stat, self.quantile, dim=0), -1
                )
                maxs = self.op.unsqueeze(
                    self.op.quantile(fit_stat, 1 - self.quantile, dim=0), -1
                )
                min_max = self.op.cat([mins, maxs], dim=-1)
                min_maxs.append(min_max)

            self.min_maxs[cls] = min_maxs

        devnorm = []
        for cls in self._classes:
            min_maxs = []
            for min_max in self.min_maxs[cls]:
                min_maxs.append(
                    self.op.stack([min_max for i in range(val_stats[0].shape[0])])
                )
            devnorm.append(
                [
                    float(self.op.mean(dev))
                    for dev in self._deviation(val_stats, min_maxs)
                ]
            )
        self.devnorm = np.mean(np.array(devnorm), axis=0)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the aggregation of deviations from quantiles of in-distribution channel-wise
        correlations evaluate for each layer, power of gram matrices, and class.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        tensor_stats, _ = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )

        _, logits = self.feature_extractor.predict_tensor(inputs)
        preds = self.op.convert_to_numpy(self.op.argmax(logits, dim=1))

        # We stack the min_maxs for each class depending on the prediction for each
        # samples
        min_maxs = []
        for i in range(len(tensor_stats)):
            min_maxs.append(self.op.stack([self.min_maxs[label][i] for label in preds]))

        tensor_dev = self._deviation(tensor_stats, min_maxs)
        score = self.op.mean(
            self.op.cat(
                [
                    self.op.unsqueeze(tensor_dev_l, dim=0) / devnorm_l
                    for tensor_dev_l, devnorm_l in zip(tensor_dev, self.devnorm)
                ]
            ),
            dim=0,
        )
        return self.op.convert_to_numpy(score)

    def _deviation(
        self, stats: List[TensorType], min_maxs: List[TensorType]
    ) -> List[TensorType]:
        """Compute the deviation wrt quantiles (min/max) for feature_maps

        Args:
            stats (TensorType): The list of gram matrices (stacked power-wise)
                for which we want to compute the deviation.
            min_maxs (TensorType): The quantiles (tensorised) to compute the deviation
                against.

        Returns:
            List(TensorType): A list with one element per layer containing a tensor of
                per-sample deviation.
        """
        deviation = []
        for stat, min_max in zip(stats, min_maxs):
            where_min = self.op.where(stat < min_max[..., 0], 1.0, 0.0)
            where_max = self.op.where(stat > min_max[..., 1], 1.0, 0.0)
            deviation_min = (
                (min_max[..., 0] - stat)
                / (self.op.abs(min_max[..., 0]) + 1e-6)
                * where_min
            )
            deviation_max = (
                (stat - min_max[..., 1])
                / (self.op.abs(min_max[..., 1]) + 1e-6)
                * where_max
            )
            deviation.append(self.op.sum(deviation_min + deviation_max, dim=(1, 2)))
        return deviation

    def _stat(self, feature_map: TensorType) -> TensorType:
        """Compute the correlation map (stat) for a given feature map. The values
        for each power of gram matrix are contained in the same tensor

        Args:
            feature_map (TensorType): The input feature_map

        Returns:
            TensorType: The stacked gram matrices power-wise.
        """
        fm_s = feature_map.shape
        stat = []
        for p in self.orders:
            feature_map_p = feature_map**p
            # construct the Gram matrix
            if len(fm_s) == 2:
                # build gram matrix for feature map of shape [dim_dense_layer, 1]
                feature_map_p = self.op.einsum(
                    "bi,bj->bij", feature_map_p, feature_map_p
                )
            elif len(fm_s) >= 3:
                # flatten the feature map
                if self.backend == "tensorflow":
                    feature_map_p = self.op.reshape(
                        self.op.einsum("i...j->ij...", feature_map_p),
                        (fm_s[0], fm_s[-1], -1),
                    )
                else:
                    feature_map_p = self.op.reshape(
                        feature_map_p, (fm_s[0], fm_s[1], -1)
                    )
                feature_map_p = self.op.matmul(
                    feature_map_p, self.op.permute(feature_map_p, (0, 2, 1))
                )
            feature_map_p = self.op.sign(feature_map_p) * (
                self.op.abs(feature_map_p) ** (1 / p)
            )
            # get the lower triangular part of the matrix
            feature_map_p = self.op.tril(feature_map_p)
            # directly sum row-wise (to limit computational burden)
            feature_map_p = self.op.sum(feature_map_p, dim=2)
            # stat.append(self.op.t(feature_map_p))
            stat.append(feature_map_p)
        stat = self.op.stack(stat, 1)
        return stat

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
        return False
