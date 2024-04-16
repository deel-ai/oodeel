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


class SHE(OODBaseDetector):
    """
    "Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization
    with Modern Hopfield Energy"
    [link](https://openreview.net/forum?id=KkazG4lgKL)

    This method first computes the mean of the internal layer representation of ID data
    for each ID class. This mean is seen as the average of the ID activation patterns
    as defined in the original paper.
    The method then returns the maximum value of the dot product between the internal
    layer representation of the input and the average patterns, which is a simplified
    version of Hopfield energy as defined in the original paper.

    Remarks:
    *   An input perturbation is applied in the same way as in Mahalanobis score
    *   The original paper only considers the penultimate layer of the neural
    network, while we aggregate the results of multiple layers after normalizing by
    the dimension of each vector (the activation vector for dense layers, and the
    average pooling of the feature map for convolutional layers).
    """

    def __init__(
        self,
        eps: float = 0.0014,
    ):
        super().__init__()
        self.eps = eps
        self.postproc_fns = None

    def _postproc_feature_maps(self, feature_map):
        if len(feature_map.shape) > 2:
            feature_map = self.op.avg_pool_2d(feature_map)
        return self.op.flatten(feature_map)

    def _fit_to_dataset(
        self,
        fit_dataset: Union[TensorType, DatasetType],
    ) -> None:
        """
        Compute the means of the input dataset in the activation space of the selected
        layers. The means are computed for each class in the dataset.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): input dataset (ID) to
                construct the index with.
            ood_dataset (Union[TensorType, DatasetType]): OOD dataset to tune the
                aggregation coefficients.
        """
        self.postproc_fns = [
            self._postproc_feature_maps
            for i in range(len(self.feature_extractor.feature_layers_id))
        ]

        features, infos = self.feature_extractor.predict(
            fit_dataset, postproc_fns=self.postproc_fns
        )

        labels = infos["labels"]

        # unique sorted classes
        self._classes = np.sort(np.unique(self.op.convert_to_numpy(labels)))

        self._mus = list()
        for feature in features:
            mus_f = list()
            for cls in self._classes:
                indexes = self.op.equal(labels, cls)
                _features_cls = feature[indexes]
                mus_f.append(
                    self.op.unsqueeze(self.op.mean(_features_cls, dim=0), dim=0)
                )
            self._mus.append(self.op.permute(self.op.cat(mus_f), (1, 0)))

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the aggregation of neural mean discrepancies from different layers.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        inputs_p = self._input_perturbation(inputs)
        features, logits = self.feature_extractor.predict_tensor(
            inputs_p, postproc_fns=self.postproc_fns
        )

        scores = self._get_she_output(features)

        return -self.op.convert_to_numpy(scores)

    def _get_she_output(self, features):
        scores = None
        for feature, mus_f in zip(features, self._mus):
            she = self.op.matmul(self.op.squeeze(feature), mus_f) / feature.shape[1]
            she = self.op.max(she, dim=1)
            scores = she if scores is None else she + scores
        return scores

    def _input_perturbation(self, inputs: TensorType) -> TensorType:
        """
        Apply small perturbation on inputs to make the in- and out- distribution
        samples more separable.

        Args:
            inputs (TensorType): input samples

        Returns:
            TensorType: Perturbed inputs
        """

        def __loss_fn(inputs: TensorType) -> TensorType:
            """
            Loss function for the input perturbation.

            Args:
                inputs (TensorType): input samples

            Returns:
                TensorType: loss value
            """
            # extract features
            out_features, _ = self.feature_extractor.predict(
                inputs, detach=False, postproc_fns=self.postproc_fns
            )
            # get mahalanobis score for the class maximizing it
            she_score = self._get_she_output(out_features)
            log_probs_f = self.op.log(she_score)
            return self.op.mean(log_probs_f)

        # compute gradient
        gradient = self.op.gradient(__loss_fn, inputs)
        gradient = self.op.sign(gradient)

        inputs_p = inputs - self.eps * gradient
        return inputs_p

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
