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
from ..types import TensorType
from ..types import Tuple
from oodeel.methods.mahalanobis import Mahalanobis


class RMDS(Mahalanobis):
    """
    "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection"
    https://arxiv.org/abs/2106.09022

    Args:
        eps (float): magnitude for gradient based input perturbation.
            Defaults to 0.02.
    """

    def __init__(self, eps: float = 0.002):
        super().__init__(eps=eps)

    def _fit_to_dataset(self, fit_dataset: DatasetType) -> None:
        """
        Constructs the pear class means and the covariance matrice,
        as well as the background mean and covariance matrice,
        from ID data "fit_dataset".
        The means and pseudo-inverses of the covariance matrices
        will be used for RMDS score computation.

        Args:
            fit_dataset (Union[TensorType, DatasetType]): input dataset (ID)
        """
        # means and pseudo-inverse of the mean convariance matrice from Mahalanobis method
        super()._fit_to_dataset(fit_dataset)

        # extract features
        features, _ = self.feature_extractor.predict(fit_dataset)

        # comput background mu and cov
        _features_bg = self.op.flatten(features[0])
        mu_bg = self.op.mean(_features_bg, dim=0)
        _zero_f_bg = _features_bg - mu_bg
        cov_bg = self.op.matmul(self.op.t(_zero_f_bg), _zero_f_bg) / _zero_f_bg.shape[0]

        # background mu and pseudo-inverse of the mean covariance matrices
        self._mu_bg = mu_bg
        self._pinv_cov_bg = self.op.pinv(cov_bg)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes an OOD score for input samples "inputs" based on the RMDS
        distance with respect to the closest class-conditional Gaussian distribution,
        and the background distribution.

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
        gaussian_score_corrected = self.op.max(
            gaussian_score_bg - gaussian_score_p, dim=1
        )
        return -self.op.convert_to_numpy(gaussian_score_corrected)

    def _background_score(self, out_features: TensorType) -> TensorType:
        """
        Mahalanobis distance-based background score. For each test sample, it computes
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
            self.op.matmul(self.op.matmul(zero_f, self._pinv_cov_bg), self.op.t(zero_f))
        )
        gaussian_score = self.op.reshape(log_probs_f, (-1, 1))
        return gaussian_score
