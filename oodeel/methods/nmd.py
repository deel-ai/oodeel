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


class NeuralMeanDiscrepancy(OODBaseDetector):
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
    ):
        super().__init__()
        self.postproc_fns = None
        self.means = None

    def _fit_to_dataset(
        self,
        fit_dataset: Union[TensorType, DatasetType],
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
            fit_dataset, postproc_fns=self.postproc_fns
        )
        self.means = self.op.unsqueeze(
            self.op.cat([self.op.mean(fs, dim=0) for fs in fit_stats]), dim=0
        )

        # TODO self.coeffs

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

        # We stack the min_maxs for each class depending on the prediction for each
        # samples
        features = self.op.cat(tensor_stats, dim=1) - self.means

    def _stat(self, feature_map: TensorType) -> TensorType:
        """Compute the correlation map (stat) for a given feature map. The values
        for each power of gram matrix are contained in the same tensor

        Args:
            feature_map (TensorType): The input feature_map

        Returns:
            TensorType: The stacked gram matrices power-wise.
        """
        fm_s = feature_map.shape
        if len(fm_s) == 2:
            return self.op.unsqueeze(self.op.mean(feature_map, dim=1), dim=1)
        elif len(fm_s) == 3:
            if self.backend == "tensorflow":
                return self.op.mean(feature_map, dim=(1, 2))
            elif self.backend == "pytorch":
                return self.op.mean(feature_map, dim=(2, 3))
        elif len(fm_s) >= 4:
            raise NotImplementedError(
                "Feature map with more than 3 dimensions not implemented"
            )

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
