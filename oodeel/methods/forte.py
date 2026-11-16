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
from typing import get_args
from typing import List
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from ..types import DatasetType
from ..types import ItemType
from ..types import TensorType
from ..types import Tuple
from ..types import Union
from .base import FeatureBasedDetector

GMM_COMPONENTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KDE_BANDWIDTHS = ("scott", "silverman")
OCSVM_NUS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
# number of points whose pairwise distances are held in memory at once
CHUNK_SIZE = 2048


def knn_radii(features: np.ndarray, nearest: int) -> np.ndarray:
    """Distance of each point to its `nearest`-th nearest neighbor within
    `features`, the point itself excluded.

    Args:
        features (np.ndarray): array of shape (n, dim).
        nearest (int): rank of the neighbor.

    Returns:
        np.ndarray: radii, of shape (n,).
    """
    features = np.asarray(features, dtype=np.float64)
    n_points = features.shape[0]
    if n_points <= nearest:
        raise ValueError(
            f"Forte needs at least nearest + 1 = {nearest + 1} points in a sample to "
            + f"compute its nearest neighbor radii, got {n_points}."
        )
    radii = np.empty(n_points)
    for start in range(0, n_points, CHUNK_SIZE):
        chunk = slice(start, start + CHUNK_SIZE)
        dist = euclidean_distances(features[chunk], features)
        # index `nearest` of the sorted row, since the row contains the point itself
        radii[chunk] = np.partition(dist, nearest, axis=1)[:, nearest]
    return radii


def prdc_per_point(
    reference: np.ndarray,
    sample: np.ndarray,
    nearest: int = 5,
    reference_radii: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-point precision, recall, density and coverage of the points of `sample`
    with respect to `reference`.

    Recall and coverage rely on the nearest neighbor radii of the points of `sample`
    computed within `sample`, so the statistics of a point depend on the sample it
    belongs to.

    Args:
        reference (np.ndarray): reference features, of shape (n_ref, dim).
        sample (np.ndarray): evaluated features, of shape (n, dim).
        nearest (int): number of nearest neighbors. Defaults to 5.
        reference_radii (Optional[np.ndarray]): precomputed `knn_radii` of
            `reference`. Defaults to None.

    Returns:
        np.ndarray: array of shape (n, 4) with columns (recall, density, precision,
            coverage).
    """
    reference = np.asarray(reference, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    if reference_radii is None:
        reference_radii = knn_radii(reference, nearest)
    sample_radii = knn_radii(sample, nearest)

    n_ref = reference.shape[0]
    stats = np.empty((sample.shape[0], 4))
    for start in range(0, sample.shape[0], CHUNK_SIZE):
        chunk = slice(start, start + CHUNK_SIZE)
        dist = euclidean_distances(reference, sample[chunk])
        in_reference_balls = dist < reference_radii[:, None]
        stats[chunk, 0] = (dist < sample_radii[chunk]).sum(axis=0) / n_ref
        stats[chunk, 1] = in_reference_balls.sum(axis=0) / (nearest * n_ref)
        stats[chunk, 2] = in_reference_balls.any(axis=0)
        stats[chunk, 3] = dist.min(axis=0) < sample_radii[chunk]
    return stats


class Forte(FeatureBasedDetector):
    """
    "Forte: Finding Outliers with Representation Typicality Estimation"
    https://openreview.net/forum?id=7XNgVPxCiA

    Forte is a two-sample test in feature space. The in-distribution features given
    to `fit` are split in two halves: a reference sample, and a second half whose
    per-point precision, recall, density and coverage (PRDC) with respect to the
    reference sample are used to fit a density model (GMM, KDE or one-class SVM).
    At test time, the PRDC statistics of the evaluated sample are computed against
    the same reference sample and scored with the density model. When several feature
    layers are used, the PRDC statistics are computed for each of them and
    concatenated before the density model, so no aggregator is involved.

    Per-point recall and coverage rely on the nearest neighbor radii of the test
    points within the evaluated sample. The score of a point therefore depends on the
    sample it is scored with: `score` gathers the features of a whole dataset before
    scoring it as one sample, and a single tensor is scored as a sample on its own.
    An evaluated sample must contain at least `nearest + 1` points.

    Args:
        nearest (int): number of nearest neighbors used for the PRDC statistics.
            Defaults to 5.
        density_model (str): density model fitted on the PRDC statistics, among
            "gmm", "kde" and "ocsvm". Defaults to "gmm".
        seed (int): random seed of the reference / fit split and of the GMM.
            Defaults to 0.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(
        self,
        nearest: int = 5,
        density_model: str = "gmm",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if density_model not in ("gmm", "kde", "ocsvm"):
            raise ValueError(
                "density_model must be one of 'gmm', 'kde' or 'ocsvm', "
                + f"got {density_model}"
            )
        self.nearest = nearest
        self.density_model = density_model
        self.seed = seed
        self._references: List[Tuple[np.ndarray, np.ndarray]] = []
        self._fit_stats: List[np.ndarray] = []
        self._density = None

    # === Public API (override of score) ===
    def score(
        self,
        dataset: Union[ItemType, DatasetType],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs". A dataset is scored as one
        evaluated sample: the features of all its batches are gathered before
        computing the scores.

        Args:
            dataset (Union[ItemType, DatasetType]): dataset or tensors to score
            verbose (bool): if True, display a progress bar. Defaults to False.

        Returns:
            tuple: scores or list of scores (depending on the input) and a dictionary
                containing logits and labels.
        """
        if not isinstance(dataset, get_args(DatasetType)):
            return super().score(dataset, verbose=verbose)

        assert self.feature_extractor is not None, "Call .fit() before .score()"
        feats, logits, labels = [], [], None
        for item in tqdm(dataset, desc="Scoring", disable=not verbose):
            tensor = self.data_handler.get_input_from_dataset_item(item)
            feats.append(self._extract_features(tensor))
            logits.append(self.op.convert_to_numpy(self.feature_extractor._last_logits))

            # get the label if available
            if len(item) > 1:
                labels_batch = self.op.convert_to_numpy(
                    self.data_handler.get_label_from_dataset_item(item)
                )
                labels = (
                    labels_batch if labels is None else np.append(labels, labels_batch)
                )

        scores = self._score_features([np.concatenate(f, axis=0) for f in zip(*feats)])
        info = dict(labels=labels, logits=np.concatenate(logits, axis=0))
        return scores, info

    # === Internal: Fitting logic ===
    def _fit_to_dataset(
        self,
        fit_dataset: DatasetType,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Extract features from `fit_dataset`, compute the PRDC statistics of each
        feature layer with `_fit_layer` and fit the density model on their
        concatenation.

        Args:
            fit_dataset (DatasetType): in-distribution dataset.
            verbose (bool): if True, display a progress bar. Defaults to False.
        """
        n_layers = len(self.feature_extractor.feature_layers_id)

        if self.postproc_fns is None:
            self.postproc_fns = [self.feature_extractor._default_postproc_fn] * n_layers

        feats, info = self.feature_extractor.predict(
            fit_dataset,
            postproc_fns=self.postproc_fns,
            verbose=verbose,
            return_labels=True,
            numpy_concat=True,
        )

        self._references, self._fit_stats = [], []
        for idx in range(n_layers):
            self._fit_layer(idx, self.op.convert_to_numpy(feats[idx]), info, **kwargs)

        self._fit_density_model(np.concatenate(self._fit_stats, axis=1))

    def _fit_layer(
        self,
        layer_id: int,
        layer_features: np.ndarray,
        info: dict,
        **kwargs,
    ) -> None:
        """Split the features of a layer into a reference sample and a fit sample,
        and compute the PRDC statistics of the fit sample.

        Args:
            layer_id: Index of the processed layer.
            layer_features: Feature array corresponding to that layer.
            info: Dictionary of auxiliary data (unused).
        """
        layer_features = layer_features.reshape(layer_features.shape[0], -1)
        reference, fit_sample = train_test_split(
            layer_features, test_size=0.5, random_state=self.seed
        )
        reference = np.asarray(reference, dtype=np.float64)
        radii = knn_radii(reference, self.nearest)
        self._references.append((reference, radii))
        self._fit_stats.append(
            prdc_per_point(reference, fit_sample, self.nearest, radii)
        )

    def _fit_density_model(self, stats: np.ndarray) -> None:
        """Fit the density model on PRDC statistics of in-distribution data. Its
        hyperparameter is selected on these same statistics.

        Args:
            stats (np.ndarray): PRDC statistics, of shape (n, 4 * n_layers).
        """
        best_model, best_score = None, -np.inf
        if self.density_model == "gmm":
            for n_components in GMM_COMPONENTS:
                if n_components >= stats.shape[0]:
                    continue
                model = GaussianMixture(
                    n_components=n_components, random_state=self.seed
                ).fit(stats)
                score = model.score(stats)
                if score > best_score:
                    best_model, best_score = model, score
        elif self.density_model == "kde":
            for bandwidth in KDE_BANDWIDTHS:
                model = gaussian_kde(stats.T, bw_method=bandwidth)
                score = model.logpdf(stats.T).mean()
                if score > best_score:
                    best_model, best_score = model, score
        else:
            for nu in OCSVM_NUS:
                model = OneClassSVM(kernel="rbf", gamma="auto", nu=nu).fit(stats)
                score = (model.predict(stats) == 1).mean()
                if score > best_score:
                    best_model, best_score = model, score
        self._density = best_model

    # === Internal: Scoring logic ===
    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """Compute the OOD scores of a batch, taken as one evaluated sample.

        Args:
            inputs (TensorType): input samples to score.

        Returns:
            np.ndarray: scores
        """
        return self._score_features(self._extract_features(inputs))

    def _extract_features(self, inputs: TensorType) -> List[np.ndarray]:
        """Extract the features of a batch of inputs.

        Args:
            inputs (TensorType): input samples.

        Returns:
            List[np.ndarray]: features, one array per layer.
        """
        if getattr(self, "eps", 0) > 0:
            inputs = self._input_perturbation(inputs, self.eps)

        feats, _ = self.feature_extractor.predict_tensor(
            inputs, postproc_fns=self.postproc_fns
        )
        return [self.op.convert_to_numpy(feat) for feat in feats]

    def _score_features(self, feats: List[np.ndarray]) -> np.ndarray:
        """Compute the OOD scores of one evaluated sample from its features.

        Args:
            feats (List[np.ndarray]): features of the sample, one array per layer.

        Returns:
            np.ndarray: scores, the opposite of the density model outputs.
        """
        stats = np.concatenate(
            [self._score_layer(idx, feat, {}) for idx, feat in enumerate(feats)],
            axis=1,
        )
        if self.density_model == "gmm":
            typicality = self._density.score_samples(stats)
        elif self.density_model == "kde":
            typicality = self._density.logpdf(stats.T)
        else:
            typicality = self._density.decision_function(stats)
        # the log-density may underflow to -inf, keep the scores finite
        return np.clip(-typicality, -np.finfo(float).max, np.finfo(float).max)

    def _score_layer(
        self,
        layer_id: int,
        layer_features: TensorType,
        info: dict,
        fit: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Compute the PRDC statistics of one evaluated sample for a single feature
        layer.

        Args:
            layer_id: Index of the processed layer.
            layer_features: Features of the sample for this layer.
            info: Dictionary of auxiliary data (unused).
            fit: Unused.

        Returns:
            np.ndarray: array of shape (n, 4) with columns (recall, density,
                precision, coverage).
        """
        reference, radii = self._references[layer_id]
        layer_features = np.asarray(layer_features)
        layer_features = layer_features.reshape(layer_features.shape[0], -1)
        return prdc_per_point(reference, layer_features, self.nearest, radii)

    # === Properties ===
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
