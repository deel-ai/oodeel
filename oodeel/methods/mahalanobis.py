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
from enum import auto
from enum import IntEnum
from typing import Dict
from typing import List

import numpy as np
from numpy.typing import ArrayLike
from sklearn import covariance
from sklearn import preprocessing

from oodeel.methods.base import OODModel


class MahalanobisMode(IntEnum):
    default = auto()
    sklearn = auto()


class Mahalanobis(OODModel):
    """
    Implementation from https://github.com/pokaxpoka/deep_Mahalanobis_detector
    """

    def __init__(
        self,
        input_processing_magnitude: float = 0.0,
        output_layers_id: List[int] = [-2],
        mode: str = "default",
    ):
        super(Mahalanobis, self).__init__(output_layers_id=output_layers_id)

        self.input_processing_magnitude = input_processing_magnitude
        self.mean_covariance: covariance.EmpiricalCovariance = (
            covariance.EmpiricalCovariance(assume_centered=True)
        )
        self.by_label_preprocessing: Dict[int, preprocessing.StandardScaler] = dict()
        self.mode: MahalanobisMode = MahalanobisMode[mode]

    def _score_tensor(self, inputs: ArrayLike):
        if self.mode == MahalanobisMode.sklearn:
            return self._score_tensor_w_sklearn(inputs)
        return self._score_tensor_default(inputs)

    def _score_tensor_w_sklearn(self, inputs: ArrayLike):
        """
        Uses sklearn.covariance.EmpiricalCovariance mahalanobis distance implementation
        but does not apply input processing
        """

        features = self.op.convert_to_numpy(self.feature_extractor.predict(inputs))

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        return np.min(
            np.stack(
                [
                    self.mean_covariance.mahalanobis(
                        self.by_label_preprocessing[lbl].transform(features)
                    )
                    for lbl in self.by_label_preprocessing.keys()
                ],
                axis=0,
            ),
            axis=0,
        )

    def _score_tensor_default(self, inputs: ArrayLike):
        """
        Code was taken from https://github.com/pokaxpoka/deep_Mahalanobis_detector with
        minimal changes
        """

        # Make sure that gradient will be retained
        out_features = self.feature_extractor.predict(inputs)
        # Flatten the features to 2D (n_batch, n_features)
        out_features = self.op.flatten(out_features)
        gaussian_score = self.mahalanobis_score(out_features)

        # Input preprocessing
        sample_pred = gaussian_score.max(1)[1]

        means = self.op.stack(
            [
                self.get_mean_by_label(list(self.by_label_preprocessing.keys())[s])
                for s in sample_pred
            ],
            dim=0,
        )

        gradient = self.op.gradient(self.loss_fn, inputs, means=means)
        gradient = self.op.sign(gradient)

        tempInputs = self.op.add(inputs, -self.input_processing_magnitude, gradient)
        noise_out_features = self.feature_extractor.predict(tempInputs)
        noise_out_features = self.op.flatten(noise_out_features)

        noise_gaussian_score = self.mahalanobis_score(noise_out_features)
        noise_gaussian_score, _ = self.op.max(noise_gaussian_score, dim=1)

        return self.op.convert_to_numpy(noise_gaussian_score)

    def mahalanobis_score(self, out_features):
        _precision = self.get_precision()
        # compute Mahalanobis score
        # for each class, remove the mean and compute Mahalanobis distance
        gaussian_scores = list()
        for i, lbl in enumerate(self.by_label_preprocessing.keys()):
            batch_sample_mean = self.get_mean_by_label(lbl)
            zero_f = out_features - batch_sample_mean
            term_gau = -0.5 * self.op.diag(
                self.op.matmul(
                    self.op.matmul(zero_f, _precision), self.op.transpose(zero_f)
                )
            )
            gaussian_scores.append(self.op.reshape(term_gau, (-1, 1)))
        gaussian_score = self.op.cat(gaussian_scores, 1)
        return gaussian_score

    def get_precision(self):
        return self.op.from_numpy(self.mean_covariance.covariance_)

    def get_mean_by_label(self, lbl):
        return self.op.from_numpy(self.by_label_preprocessing[lbl].mean_)

    def loss_fn(self, inputs: ArrayLike, means: np.ndarray):
        _out_features = self.feature_extractor.predict(inputs, detach=False)
        # Flatten the features to 2D (n_batch, n_features)
        _out_features = self.op.flatten(_out_features)
        _zero_f = _out_features - self.op.from_numpy(means)
        pure_gau = -0.5 * self.op.diag(
            self.op.matmul(
                self.op.matmul(_zero_f, self.op.from_numpy(self.get_precision())),
                self.op.transpose(_zero_f),
            )
        )
        return self.op.mean(-pure_gau)

    def _fit_to_dataset(self, fit_dataset: ArrayLike):
        # Store feature sets by label
        features_by_label: Dict[int, List[np.ndarray]] = dict()

        for batch in fit_dataset:
            images, labels = batch
            features = self.feature_extractor.predict(images)
            for lbl in np.unique(self.op.convert_to_numpy(labels)):
                if lbl not in features_by_label.keys():
                    features_by_label[lbl] = list()
                _feat_np = self.op.convert_to_numpy(features[labels == lbl])
                _feat_np = _feat_np.reshape(_feat_np.shape[0], -1)
                features_by_label[lbl].append(_feat_np)
        for lbl in features_by_label.keys():
            features_by_label[lbl] = np.vstack(features_by_label[lbl])

        # Remove mean for each label and compute covariance
        by_label_preprocessing = dict()
        by_label_covariance = dict()
        for lbl in features_by_label.keys():
            ss = preprocessing.StandardScaler(with_mean=True, with_std=False)
            ss.fit(features_by_label[lbl])

            ec = covariance.EmpiricalCovariance(assume_centered=True)
            ec.fit(ss.transform(features_by_label[lbl]))

            by_label_preprocessing[lbl] = ss
            by_label_covariance[lbl] = ec

        # Take the mean of the per class covariances
        self.mean_covariance = covariance.EmpiricalCovariance(assume_centered=True)
        self.mean_covariance._set_covariance(
            np.mean(
                np.stack(
                    [
                        by_label_covariance[lbl].covariance_
                        for lbl in by_label_covariance.keys()
                    ],
                    axis=0,
                ),
                axis=0,
            )
        )
        labels = list(features_by_label.keys())
        self.mean_covariance.location_ = np.zeros(features_by_label[labels[0]].shape[1])

        self.by_label_preprocessing = by_label_preprocessing
        self.by_label_covariance = by_label_covariance
