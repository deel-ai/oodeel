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
from sklearn import covariance
from sklearn import preprocessing

from ..types import DatasetType
from ..types import List
from ..types import TensorType
from ..types import Union
from oodeel.methods.base import OODModel


class Mahalanobis(OODModel):
    """
    "A Simple Unified Framework for Detecting Out-of-Distribution Samples and
    Adversarial Attacks"
    https://arxiv.org/abs/1807.03888

    Args:
        eps (float): magnitude for gradient based input perturbation.
            Defaults to 0.02.
        output_layers_id (List[int]): feature space on which to compute mahalanobis
            distance. Defaults to [-2].
    """

    def __init__(
        self,
        eps: float = 0.02,
        output_layers_id: List[int] = [-2],
    ):
        super(Mahalanobis, self).__init__(output_layers_id=output_layers_id)
        self.eps = eps

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]):
        """
        Constructs the mean covariance matrix from ID data "fit_dataset", whose
        pseudo-inverse will be used for mahalanobis distance computation.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): input dataset (ID)
        """
        # Store feature sets by label
        features_by_label = dict()
        for batch in fit_dataset:
            images, labels = batch
            # if one hot encoded labels, take the argmax
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = self.op.argmax(labels.reshape(labels.shape[0], -1), 1)
            labels = self.op.convert_to_numpy(labels)

            # extract features
            features = self.feature_extractor.predict(images)

            # store features by label
            for lbl in labels:
                if lbl not in features_by_label.keys():
                    features_by_label[lbl] = list()
                _feat_np = self.op.convert_to_numpy(features[labels == lbl])
                _feat_np = _feat_np.reshape(_feat_np.shape[0], -1)
                features_by_label[lbl].append(_feat_np)
        for lbl in features_by_label.keys():
            features_by_label[lbl] = np.vstack(features_by_label[lbl])

        # store labels indexes
        self._labels_indexes = list(features_by_label.keys())

        # compute centered covariances cond. to each class distribution
        mus = dict()
        covs = dict()
        for lbl in self._labels_indexes:
            ss = preprocessing.StandardScaler(with_mean=True, with_std=False)
            ss.fit(features_by_label[lbl])

            ec = covariance.EmpiricalCovariance(assume_centered=True)
            ec.fit(ss.transform(features_by_label[lbl]))

            mus[lbl] = ss.mean_
            covs[lbl] = ec.covariance_

        # Take the mean of the per class covariances
        mean_covariance = covariance.EmpiricalCovariance(assume_centered=True)
        mean_covariance._set_covariance(
            np.mean(np.stack(list(covs.values()), axis=0), axis=0)
        )

        # store centers and pseudo inverse of mean covariance matrix
        self._mus = mus
        self._pinv_cov = self.op.from_numpy(mean_covariance.precision_)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on the mahalanobis
        distance with respect to the closest class-conditional Gaussian distribution.

        Args:
            inputs (TensorType): input samples

        Returns:
            np.ndarray: ood scores
        """
        # input preprocessing (perturbation)
        if self.eps > 0:
            inputs_p = self._input_perturbation(inputs)

        # mahalanobis score on perturbed inputs
        features_p = self.feature_extractor.predict(inputs_p)
        features_p = self.op.flatten(features_p)
        gaussian_score_p = self._mahalanobis_score(features_p)

        # take the highest score for each sample
        gaussian_score_p = self.op.max(gaussian_score_p, dim=1)
        return -self.op.convert_to_numpy(gaussian_score_p)

    def _input_perturbation(self, inputs: TensorType) -> TensorType:
        """
        Apply small perturbation on inputs to make the in- and out- distribution
        samples more separable.
        See original paper for more information (section 2.2)
        https://arxiv.org/abs/1807.03888

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
                TensorType: loss function
            """
            # extract features
            _out_features = self.feature_extractor.predict(inputs, detach=False)
            _out_features = self.op.flatten(_out_features)
            # get mahalanobis score for the class maximizing it
            gaussian_score = self._mahalanobis_score(_out_features)
            pure_gau = self.op.max(gaussian_score, dim=1)
            return self.op.mean(-pure_gau)

        # compute gradient
        gradient = self.op.gradient(__loss_fn, inputs)
        gradient = self.op.sign(gradient)

        inputs_p = inputs - self.eps * gradient
        return inputs_p

    def _mahalanobis_score(self, out_features: TensorType) -> TensorType:
        """
        Mahalanobis distance-based confidence score. For each test sample, it computes
        the Mahalanobis distance with respect to the every class-conditional Gaussian
        distributions.

        Args:
            out_features (TensorType): test samples features

        Returns:
            TensorType: confidence scores (conditionally to each class)
        """
        gaussian_scores = list()
        # compute scores conditionally to each class
        for lbl in self._labels_indexes:
            mus = self._get_mus_from_labels(lbl)
            term_gau = self._log_prob_mahalanobis(out_features, mus)
            gaussian_scores.append(self.op.reshape(term_gau, (-1, 1)))
        # concatenate scores
        gaussian_score = self.op.cat(gaussian_scores, 1)
        return gaussian_score

    def _log_prob_mahalanobis(
        self, features: TensorType, mus: TensorType
    ) -> TensorType:
        """
        Compute the log of the probability densities of some observations (assuming a
        normal distribution) using the mahalanobis distance with respect to some
        classconditional distributions.

        Args:
            features (TensorType): observations features
            mus (TensorType): centers of classconditional distributions

        Returns:
            TensorType: log probability tensor
        """
        zero_f = features - mus
        term_gau = -0.5 * self.op.diag(
            self.op.matmul(
                self.op.matmul(zero_f, self._pinv_cov), self.op.transpose(zero_f)
            )
        )
        return term_gau

    def _get_mus_from_labels(self, lbl: int) -> TensorType:
        return self.op.from_numpy(self._mus[lbl])
