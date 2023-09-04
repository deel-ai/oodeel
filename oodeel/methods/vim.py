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
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from ..types import DatasetType
from ..types import TensorType
from ..types import Tuple
from ..types import Union
from .base import OODBaseDetector


class VIM(OODBaseDetector):
    """
    Compute the Virtual Matching Logit (VIM) score.
    https://arxiv.org/abs/2203.10807

    This score combines the energy score with a PCA residual score.

    The energy score is the logarithm of the sum of exponential of logits.
    The PCA residual score is based on the projection on residual dimensions for
    principal component analysis.
        Residual dimensions are the eigenvectors corresponding to the least eignevalues
        (least variance).
        Intuitively, this score method assumes that feature representations of ID data
        occupy a low dimensional affine subspace $P+c$ of the feature space.
        Specifically, the projection of ID data translated by $-c$ on the
        orthognoal complement $P^{\\perp}$ is expected to have small norm.
        It allows to detect points whose feature representation lie far from the
        identified affine subspace, namely those points $x$ such that the
        projection on $P^{\\perp}$ of $x-c$ has large norm.

    Args:
        princ_dims (Union[int, float]): number of principal dimensions of in
            distribution features to consider. If an int, must be less than the
            dimension of the feature space.
            If a float, it must be in [0,1), it represents the ratio of
            explained variance to consider to determine the number of principal
            components. Defaults to 0.99.
        pca_origin (str): either "pseudo" for using $W^{-1}b$ where $W^{-1}$ is
            the pseudo inverse of the final linear layer applied to bias term
            (as in the VIM paper), or "center" for using the mean of the data in
            feature space. Defaults to "center".
    """

    def __init__(
        self,
        princ_dims: Union[int, float] = 0.99,
        pca_origin: str = "pseudo",
    ):
        super().__init__()
        self._princ_dim = princ_dims
        self.pca_origin = pca_origin

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]) -> None:
        """
        Computes principal components of feature representations and store the residual
        eigenvectors.
        Computes a scaling factor constant :math:'\alpha' such that the average scaled
        residual score (on train) is equal to the average maximum logit score (MLS)
        score.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        # extract features from fit dataset
        all_features_train, info = self.feature_extractor.predict(fit_dataset)
        features_train = all_features_train
        logits_train = info["logits"]
        features_train = self.op.flatten(features_train)
        self.feature_dim = features_train.shape[1]
        logits_train = self.op.convert_to_numpy(logits_train)

        # get distribution center for pca projection
        if self.pca_origin == "center":
            self.center = self.op.mean(features_train, dim=0)
        elif self.pca_origin == "pseudo":
            # W, b = self.feature_extractor.get_weights(
            #    self.feature_extractor.feature_layers_id[0]
            # )
            W, b = self.feature_extractor.get_weights(-1)
            W, b = self.op.from_numpy(W), self.op.from_numpy(b.reshape(-1, 1))
            _W = self.op.transpose(W) if self.backend == "tensorflow" else W
            self.center = -self.op.reshape(self.op.matmul(self.op.pinv(_W), b), (-1,))
        else:
            raise NotImplementedError(
                'only "center" and "pseudo" are available for argument "pca_origin"'
            )

        # compute eigvalues and eigvectors of empirical covariance matrix
        centered_features = features_train - self.center
        emp_cov = (
            self.op.matmul(self.op.transpose(centered_features), centered_features)
            / centered_features.shape[0]
        )
        eig_vals, eigen_vectors = self.op.eigh(emp_cov)
        self.eig_vals = self.op.convert_to_numpy(eig_vals)

        # get number of residual dims for pca projection
        if isinstance(self._princ_dim, int):
            assert self._princ_dim < self.feature_dim, (
                f"if 'princ_dims'(={self._princ_dim}) is an int, it must be less than "
                "feature space dimension ={self.feature_dim})"
            )
            self.res_dim = self.feature_dim - self._princ_dim
            self._princ_dim = self._princ_dim
        elif isinstance(self._princ_dim, float):
            assert (
                0 <= self._princ_dim and self._princ_dim < 1
            ), f"if 'princ_dims'(={self._princ_dim}) is a float, it must be in [0,1)"
            explained_variance = np.cumsum(
                np.flip(self.eig_vals) / np.sum(self.eig_vals)
            )
            self._princ_dim = np.where(explained_variance > self._princ_dim)[0][0]
            self.res_dim = self.feature_dim - self._princ_dim

        # projector on residual space
        self.res = eigen_vectors[:, : self.res_dim]  # asc. order with eigh

        # compute residual score on training data
        train_residual_scores = self._compute_residual_score_tensor(features_train)
        # compute MLS on training data
        train_mls_scores = np.max(logits_train, axis=-1)
        # compute scaling factor
        self.alpha = np.mean(train_mls_scores) / np.mean(train_residual_scores)

    def _compute_residual_score_tensor(self, features: TensorType) -> np.ndarray:
        """
        Computes the norm of the residual projection in the feature space.

        Args:
            features: input samples to score

        Returns:
            np.ndarray: scores
        """
        res_coordinates = self.op.matmul(features - self.center, self.res)
        # taking the norm of the coordinates, which amounts to the norm of
        # the projection since the eigenvectors form an orthornomal basis
        res_norm = self.op.norm(res_coordinates, dim=-1)
        return self.op.convert_to_numpy(res_norm)

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes the VIM score for input samples "inputs" as the sum of the energy
        score and a scaled (PCA) residual norm in the feature space.

        Args:
            inputs: input samples to score

        Returns:
            Tuple[np.ndarray]: scores, logits
        """
        # extract features
        features, logits = self.feature_extractor.predict_tensor(inputs)
        features = self.op.flatten(features)
        # vim score
        res_scores = self._compute_residual_score_tensor(features)
        logits = self.op.convert_to_numpy(logits)
        energy_scores = logsumexp(logits, axis=-1)
        scores = -self.alpha * res_scores + energy_scores
        return -np.array(scores)

    def plot_spectrum(self) -> None:
        """
        Plot cumulated explained variance wrt the number of principal dimensions.
        """
        cumul_explained_variance = np.cumsum(self.eig_vals)[::-1]
        plt.plot(cumul_explained_variance / np.max(cumul_explained_variance))
        plt.axvline(
            x=self._princ_dim,
            color="r",
            linestyle="--",
            label=f"princ_dims = {self._princ_dim} ",
        )
        plt.legend()
        plt.ylabel("Residual explained variance")
        plt.xlabel("Number of principal dimensions")

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
