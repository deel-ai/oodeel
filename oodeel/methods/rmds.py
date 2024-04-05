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

from oodeel.methods.base import OODBaseDetector
from ..types import DatasetType
from ..types import TensorType
from ..types import Tuple


class RMDS(OODBaseDetector):
    """
    "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection"
    https://arxiv.org/abs/2106.09022

    Args:
        eps (float): magnitude for gradient based input perturbation.
            Defaults to 0.02.
    """

    def __init__(
        self,
        eps: float = 0.002,
    ):
        super(RMDS, self).__init__()
        self.eps = eps

    def _fit_to_dataset(self, fit_dataset: DatasetType) -> None:
        """
        Constructs the mean and covariance matrix from ID data "fit_dataset", whose
        pseudo-inverse will be used for mahalanobis distance computation.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): input dataset (ID)
        """
        # extract features and labels
        features, infos = self.feature_extractor.predict(fit_dataset)
        labels = infos["labels"]

        # unique sorted classes
        self._classes = np.sort(np.unique(self.op.convert_to_numpy(labels)))

        # compute mus and covs
        mus = dict()
        covs = dict()
        for cls in self._classes:
            indexes = self.op.equal(labels, cls)
            _features_cls = self.op.flatten(features[0][indexes])
            mus[cls] = self.op.mean(_features_cls, dim=0)
            _zero_f_cls = _features_cls - mus[cls]
            cov_cls = (
                self.op.matmul(self.op.t(_zero_f_cls), _zero_f_cls)
                / _zero_f_cls.shape[0]
            )
            if mean_cov is None:
                mean_cov = (len(_features_cls) / len(features)) * cov_cls
            else:
                mean_cov += (len(_features_cls) / len(features)) * cov_cls

        # comput background mu and cov
        _features_bg = self.op.flatten(features[0])
        mu_bg = self.op.mean(_features_bg, dim=0)
        _zero_f_bg = _features_bg - mu_bg
        cov_bg = (self.op.matmul(self.op.t(_zero_f_bg), _zero_f_bg) / _zero_f_bg.shape[0])

        # means and pseudo-inverse of the mean covariance matrices
        self._mus = mus
        self._pinv_cov = self.op.pinv(mean_cov)
        self._mu_bg = mu_bg
        self._pinv_cov_bg = self.op.pinv(cov_bg)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes an OOD score for input samples "inputs" based on the mahalanobis
        distance with respect to the closest class-conditional Gaussian distribution.

        Args:
            inputs (TensorType): input samples

        Returns:
            Tuple[np.ndarray]: scores, logits
        """
        # input preprocessing (perturbation)
        if self.eps > 0:
            inputs_p = self._input_perturbation(inputs)
        else:
            inputs_p = inputs

        # mahalanobis score on perturbed inputs
        features_p, _ = self.feature_extractor.predict_tensor(inputs_p)
        features_p = self.op.flatten(features_p[0])
        gaussian_score_p = self._mahalanobis_score(features_p)

        # background score on perturbed inputs
        gaussian_score_bg = self._background_score(features_p)

        # take the highest score for each sample
        gaussian_score_p = self.op.max(gaussian_score_p - gaussian_score_bg, dim=1)
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
                TensorType: loss value
            """
            # extract features
            out_features, _ = self.feature_extractor.predict(inputs, detach=False)
            out_features = self.op.flatten(out_features[0])
            # get mahalanobis score for the class maximizing it
            gaussian_score = self._mahalanobis_score(out_features)
            log_probs_f = self.op.max(gaussian_score, dim=1)
            return self.op.mean(-log_probs_f)

        # compute gradient
        gradient = self.op.gradient(__loss_fn, inputs)
        gradient = self.op.sign(gradient)

        inputs_p = inputs - self.eps * gradient
        return inputs_p

    def _mahalanobis_score(self, out_features: TensorType) -> TensorType:
        """
        Mahalanobis distance-based confidence score. For each test sample, it computes
        the log of the probability densities of some observations (assuming a
        normal distribution) using the mahalanobis distance with respect to every
        class-conditional distributions.

        Args:
            out_features (TensorType): test samples features

        Returns:
            TensorType: confidence scores (conditionally to each class)
        """
        gaussian_scores = list()
        # compute scores conditionally to each class
        for cls in self._classes:
            # center features wrt class-cond dist.
            mu = self._mus[cls]
            zero_f = out_features - mu
            # gaussian log prob density (mahalanobis)
            log_probs_f = -0.5 * self.op.diag(
                self.op.matmul(
                    self.op.matmul(zero_f, self._pinv_cov), self.op.t(zero_f)
                )
            )
            gaussian_scores.append(self.op.reshape(log_probs_f, (-1, 1)))
        # concatenate scores
        gaussian_score = self.op.cat(gaussian_scores, 1)
        return gaussian_score
    
    def _background_score(self, out_features: TensorType) -> TensorType:
        """
        Mahalanobis distance-based confidence score. For each test sample, it computes
        the log of the probability densities of some observations (assuming a
        normal distribution) using the mahalanobis distance with respect to the 
        background distribution.

        Args:
            out_features (TensorType): test samples features

        Returns:
            TensorType: confidence scores (with respect to the background distribution)
        """
        mu = self._mu_bg
        zero_f = out_features - mu
        # gaussian log prob density (mahalanobis)
        log_probs_f = -0.5 * self.op.diag(
            self.op.matmul(
                self.op.matmul(zero_f, self._pinv_cov_bg), self.op.t(zero_f)
            )
        )
        gaussian_score = self.op.reshape(log_probs_f, (-1, 1))
        return gaussian_score

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
