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
from scipy.linalg import eigh
from scipy.linalg import norm
from scipy.linalg import pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

from ..types import DatasetType
from ..types import List
from ..types import TensorType
from ..types import Union
from .base import OODModel


try:
    from kneed import KneeLocator
except ImportError:
    _has_kneed = False
    _kneed_not_found_err = ModuleNotFoundError(
        (
            "This function requires Kneed to be executed. Please run command "
            "`pip install kneed` 'conda install -c conda-forge kneed'"
        )
    )
else:
    _has_kneed = True


class VIM(OODModel):
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
        princ_dims: number of principal dimensions of in distribution features to
            consider.
            If an int, must be less than the dimension of the feature space.
            If a float, it must be in [0,1), it represents the ratio of explained
            variance to consider to determine the number of principal components.
            If None, the kneedle algorithm is used to determine the number of
            dimensions.
            Defaults to None.
        pca_origin: either "center" for using the mean of the data in feature space, or
            "pseudo" for using $W^{-1}b$ where $W^{-1}$ is the pseudo inverse of the final
            linear layer applied to bias term (as in the VIM paper).
            Defaults to "center".
        output_layers_id: features to use for Residual and Energy score.
            Defaults to [-2,-1] (-2 the features for PCA residual, -1 the logits with
            output_activation="linear" for Energy).
    """

    def __init__(
        self,
        princ_dims: Union[int, float] = None,
        pca_origin: str = "center",
        output_layers_id: List[int] = [-2, -1],
    ):
        super().__init__(
            output_layers_id=output_layers_id,
        )
        self._princ_dim = princ_dims
        self.pca_origin = pca_origin

    def _fit_to_dataset(self, fit_dataset: Union[TensorType, DatasetType]):
        """
        Computes principal components of feature representations and store the residual
        eigenvectors.
        Computes a scaling factor constant :math:'\alpha' such that the average scaled
        residual score (on train) is equal to the average maximum logit score (MLS)
        score.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        features_train, logits_train = self.feature_extractor.predict(fit_dataset)
        features_train = self.op.convert_to_numpy(features_train)
        logits_train = self.op.convert_to_numpy(logits_train)
        self.feature_dim = features_train.shape[1]
        if self.pca_origin == "center":
            self.center = np.mean(features_train, axis=0)
        elif self.pca_origin == "pseudo":
            W, b = self.feature_extractor.get_weights(-1)
            self.center = -np.matmul(pinv(W.T), b)
        else:
            raise NotImplementedError(
                'only "center" and "pseudo" are available for argument "pca_origin"'
            )
        ec = EmpiricalCovariance(assume_centered=True)

        ec.fit(features_train - self.center)
        # compute eigenvalues and eigenvectors of empirical covariance matrix
        eig_vals, eigen_vectors = eigh(ec.covariance_)
        # allow to use Kneedle to find res_dim
        self.eigenvalues = eig_vals

        if self._princ_dim is None:
            if not _has_kneed:
                raise _kneed_not_found_err
            # we use kneedle to look for an elbow point to set the number of principal
            # dimensions
            # we apply kneedle to the function cumsum(eigvals) which maps a dimension d
            # to the variance of the d dimensional (principal) subspace with lowest
            # variance.
            # since eigvals is non decreasing, cumsum(eigvals) is always convex and
            # increasing.
            self.kneedle = KneeLocator(
                range(len(eig_vals)),
                np.cumsum(eig_vals),
                S=1.0,
                curve="convex",
                direction="increasing",
            )
            self.res_dim = self.kneedle.elbow
            assert (
                0 < self.res_dim and self.res_dim < self.feature_dim
            ), f"Found invalid number of residual dimensions ({self.res_dim}) "
            self._princ_dim = self.feature_dim - self.res_dim
            print(
                (
                    f"Found an elbow point for {self.feature_dim-self.res_dim} principal "
                    "dimensions inside the {self.feature_dim} dimensional feature space."
                )
            )
            print(
                "You can visualize this elbow by calling the method '.plot_spectrum()' "
                "of this class"
            )
        elif isinstance(self._princ_dim, int):
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
            explained_variance = np.cumsum(np.flip(eig_vals) / np.sum(eig_vals))
            self._princ_dim = np.where(explained_variance > self._princ_dim)[0][0]
            self.res_dim = self.feature_dim - self._princ_dim

        self.res = np.ascontiguousarray(eigen_vectors[:, : self.res_dim], np.float32)

        # compute residual score on training data
        train_residual_scores = self._compute_residual_score_tensor(features_train)
        # compute MLS on training data
        train_mls_scores = np.max(logits_train, axis=-1)
        # compute scaling factor
        self.alpha = np.mean(train_mls_scores) / np.mean(train_residual_scores)

    def _compute_residual_score_tensor(self, features: TensorType) -> TensorType:
        """
        Computes the norm of the residual projection in the feature space.

        Args:
            features: input samples to score

        Returns:
            scores
        """
        res_coordinates = np.matmul(features - self.center, self.res)
        # res_coordinates = self.op.matmul(features - self.center, self.res)  # TODO
        # taking the norm of the coordinates, which amounts to the norm of
        # the projection since the eigenvectors form an orthornomal basis
        res_norm = norm(res_coordinates, axis=-1)
        # res_norm = self.op.norm(res_coordinates, dim=-1)  # TODO

        return res_norm

    def _residual_score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes the residual score for input samples "inputs".

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"
        # compute predicted features

        features = self.feature_extractor.predict(inputs)[0]
        res_scores = self._compute_residual_score_tensor(features)
        return self.op.convert_to_numpy(res_scores)

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes the VIM score for input samples "inputs" as the sum of the energy
        score and a scaled (PCA) residual norm in the feature space.

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        # compute predicted features

        features, logits = self.feature_extractor(inputs)
        features = self.op.convert_to_numpy(features)
        logits = self.op.convert_to_numpy(logits)
        res_scores = self._compute_residual_score_tensor(features)
        # res_scores = self.op.convert_to_numpy(res_scores)
        energy_scores = logsumexp(logits, axis=-1)
        scores = -self.alpha * res_scores + energy_scores
        return -np.array(scores)

    def plot_spectrum(self) -> None:
        """
        Plot cumulated explained variance wrt the number of principal dimensions.
        """
        if hasattr(self, "kneedle"):
            self.kneedle.plot_knee()
            plt.ylabel("Explained variance")
            plt.xlabel("Number of principal dimensions")
            plt.title(
                (
                    f"Found elbow at dimension {self.kneedle.elbow}\n "
                    f"{self.feature_dim-self.kneedle.elbow} principal dimensions"
                )
            )
        else:
            plt.plot(np.cumsum(self.eigenvalues))
            plt.axvline(
                x=self.res_dim,
                color="r",
                linestyle="--",
                label=f"Number of principal dimensions = {self._princ_dim} ",
            )
            plt.legend()
            plt.ylabel("Explained variance")
            plt.xlabel("Number of principal dimensions")
            plt.title("Explained variance by number of principal dimensions")
