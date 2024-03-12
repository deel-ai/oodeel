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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ..types import DatasetType
from ..types import List
from ..types import TensorType
from ..types import Union
from .base import OODBaseDetector


class NeuralMeanDiscrepancy(OODBaseDetector):
    """
    "Neural Mean Discrepancy for Efficient Out-of-Distribution Detection"
    [link](https://arxiv.org/abs/2104.11408)


    This method uses a metric called Neural Mean Discrepancy (NMD) that compares
    neural means of input examples and training data in the intermediate layers
    to identify OOD. The different values computed at each intermediate layer are
    then aggregated with coefficients that are tuned by solving a binary classification
    problem with:
        - Positive samples: the NMD values of the training data
        - Negative samples: the NMD values of proxy OOD data
                            (outlier exposure, or patch permutations)

    The binary classifier used is a Logistic Regression for simplicity.
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
        ood_dataset: Union[TensorType, DatasetType] = None,
    ) -> None:
        """
        Compute the neural mean discrepancy of the input dataset in the activation space
        of selected intermediate layers. Then, apply patch permutation to build
        proxy OOD samples and fit a binary classifier to tune the coefficients
        used to aggregate neural mean discrepancies from different layers.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): input dataset (ID) to
                construct the index with.
            ood_dataset (Union[TensorType, DatasetType]): OOD dataset to tune the
                aggregation coefficients.
        """
        self.postproc_fns = [
            self._stat for i in range(len(self.feature_extractor.feature_layers_id))
        ]

        fit_stats, _ = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns
        )

        self.means = self.op.unsqueeze(
            self.op.cat([self.op.mean(fs, dim=0) for fs in fit_stats]), dim=0
        )

        if ood_dataset is None:
            self.agg = None
            return

        ood_stats, _ = self.feature_extractor.predict(
            ood_dataset, postproc_fns=self.postproc_fns
        )

        id_data = self.op.cat(fit_stats, dim=1)
        ood_data = self.op.cat(ood_stats, dim=1)
        reg_data = self.op.cat([id_data, ood_data], dim=0)
        y = np.concatenate(
            [
                np.zeros(id_data.shape[0]),
                np.ones(ood_data.shape[0]),
            ]
        )
        self.agg = LogisticRegression()
        # self.agg = MLPClassifier(hidden_layer_sizes=(32,32,32), max_iter=60)
        self.agg.fit(self.op.convert_to_numpy(reg_data), y)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the aggregation of neural mean discrepancies from different layers.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        tensor_stats, _ = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )

        features = self.op.cat(tensor_stats, dim=1) - self.means
        features = self.op.convert_to_numpy(features)
        if self.agg is None:
            scores = np.mean(features, 1)
        else:
            scores = self.agg.predict_proba(features)[:, 1]
        return scores

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
        elif len(fm_s) == 4:
            if self.backend == "tensorflow":
                return self.op.mean(feature_map, dim=(1, 2))
            elif self.backend == "torch":
                return self.op.mean(feature_map, dim=(2, 3))
        elif len(fm_s) > 4:
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
