from enum import Enum, IntEnum, auto
from typing import List, Dict

import torch
from numpy.typing import ArrayLike
from sklearn import preprocessing, covariance
from torch.autograd import Variable

from oodeel.methods.base import OODModel
import numpy as np


# TODO Simplify and document input processing code

class MahalanobisMode(IntEnum):
    default = auto()
    sklearn = auto()


class Mahalanobis(OODModel):
    """
    Implementation from https://github.com/pokaxpoka/deep_Mahalanobis_detector
    """

    def __init__(self,
                 input_processing_magnitude: float = 0.0,
                 output_layers_id: List[int] = [-2],
                 mode: str = "default"
                 ):
        super(Mahalanobis, self).__init__(output_layers_id=output_layers_id)

        self.input_processing_magnitude = input_processing_magnitude
        self.mean_covariance: covariance.EmpiricalCovariance = covariance.EmpiricalCovariance(assume_centered=True)
        self.by_label_preprocessing: Dict[int, preprocessing.StandardScaler] = dict()
        self.mode: MahalanobisMode = MahalanobisMode[mode]

    def _score_tensor(self, inputs: ArrayLike):
        if self.mode == MahalanobisMode.sklearn:
            return self._score_tensor_w_sklearn(inputs)
        return self._score_tensor_default(inputs)

    def _score_tensor_w_sklearn(self, inputs: ArrayLike):
        """
        Uses sklearn.covariance.EmpiricalCovariance mahalanobis distance implementation but does not apply input processing
        """

        features = self.feature_extractor.predict(inputs).numpy()

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        return np.min(np.stack(
            [self.mean_covariance.mahalanobis(self.by_label_preprocessing[lbl].transform(features)) for lbl in
             self.by_label_preprocessing.keys()],
            axis=0), axis=0)

    def _score_tensor_default(self, inputs: ArrayLike):

        """
        Code was taken from https://github.com/pokaxpoka/deep_Mahalanobis_detector with minimal changes
        """

        magnitude = self.input_processing_magnitude

        # Make sure that gradient will be retained
        data = Variable(inputs, requires_grad=True)
        out_features = self.feature_extractor.predict(data, detach=False)
        # Flatten the features to 2D (n_batch, n_features)
        out_features = out_features.view(out_features.size(0), -1)
        # compute Mahalanobis score
        # for each class, remove the mean and compute Mahalanobis distance
        gaussian_score = 0
        for i, lbl in enumerate(self.by_label_preprocessing.keys()):
            # previously : batch_sample_mean = sample_mean[layer_index][i]
            batch_sample_mean = torch.from_numpy(self.by_label_preprocessing[lbl].mean_)
            #
            zero_f = out_features.data - batch_sample_mean
            # _precision = precision[layer_index]
            _precision = torch.from_numpy(self.mean_covariance.covariance_).double()
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, _precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        # previously: batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        batch_sample_mean = torch.from_numpy(
            np.stack(
                [self.by_label_preprocessing[list(self.by_label_preprocessing.keys())[s]].mean_ for s in sample_pred],
                axis=0)
        )
        #
        zero_f = out_features - Variable(batch_sample_mean)
        # previously: _precision = precision[layer_index]
        _precision = torch.from_numpy(self.mean_covariance.covariance_).double()
        #
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(_precision)), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward(retain_graph=True)

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, -magnitude, gradient)

        noise_out_features = self.feature_extractor.predict(tempInputs)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        # previously
        # for i in range(num_classes):
        #     batch_sample_mean = sample_mean[layer_index][i]
        #
        for i, lbl in enumerate(self.by_label_preprocessing.keys()):
            # previously, batch_sample_mean = sample_mean[layer_index][i]
            batch_sample_mean = torch.from_numpy(self.by_label_preprocessing[lbl].mean_).double()
            #
            zero_f = noise_out_features.data - batch_sample_mean
            # previously: _precision = precision[layer_index]
            _precision = torch.from_numpy(self.mean_covariance.covariance_).double()
            #
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, _precision), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        return noise_gaussian_score

    def _fit_to_dataset(self, fit_dataset: ArrayLike):
        # Store feature sets by label
        features_by_label: Dict[int, List[np.ndarray]] = dict()

        for batch in fit_dataset:
            images, labels = batch
            features = self.feature_extractor.predict(images)
            for lbl in np.unique(labels.numpy()):
                if not lbl in features_by_label.keys():
                    features_by_label[lbl] = list()
                _feat_np = features[labels == lbl, ::].numpy()
                _feat_np = _feat_np.reshape(_feat_np.shape[0], -1)
                features_by_label[lbl].append(
                    _feat_np
                )
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
            np.mean(np.stack([by_label_covariance[lbl].covariance_ for lbl in by_label_covariance.keys()], axis=0),
                    axis=0)
        )
        labels = list(features_by_label.keys())
        self.mean_covariance.location_ = np.zeros(features_by_label[labels[0]].shape[1])

        self.by_label_preprocessing = by_label_preprocessing
        self.by_label_covariance = by_label_covariance
