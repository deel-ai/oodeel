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
from .base import OODBaseDetector


class GEN(OODBaseDetector):
    """
    Generalized Entropy method for OOD detection.
    "GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection"
    https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.html,

    Args:
        gamma (float): parameter for the generalized entropy. Must be between 0 and 1.
            Defaults to 0.1.
        k (int): number of softmax values to keep for the entropy computation. Only the
            top-k softmax probabilities will be used. Defaults to 100.
        use_react (bool): if true, apply ReAct method by clipping penultimate
            activations under a threshold value.
        react_quantile (Optional[float]): q value in the range [0, 1] used to compute
            the react clipping threshold defined as the q-th quantile penultimate layer
            activations. Defaults to 0.8.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: int = 100,
        use_react: bool = False,
        use_scale: bool = False,
        react_quantile: float = 0.8,
        scale_percentile: float = 0.85,
        ash_percentile: float = 0.90,
    ):
        super().__init__(
            use_react=use_react,
            use_scale=use_scale,
            react_quantile=react_quantile,
            scale_percentile=scale_percentile,
            ash_percentile=ash_percentile,
        )
        self.gamma = gamma
        self.k = k

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            Tuple[np.ndarray]: scores, logits
        """

        _, logits = self.feature_extractor.predict_tensor(inputs)
        probs = self.op.softmax(logits)
        probs = self.op.convert_to_numpy(probs)
        probs = np.sort(probs)[:, -self.k :]  # Keep the k largest probabilities
        scores = np.sum(probs**self.gamma * (1 - probs) ** (self.gamma), axis=-1)
        return scores

    def _fit_to_dataset(self, fit_dataset: DatasetType) -> None:
        """
        Fits the OOD detector to fit_dataset.

        Args:
            fit_dataset: dataset to fit the OOD detector on
        """
        pass

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.

        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        return False

    @property
    def requires_internal_features(self) -> bool:
        """
        Whether an OOD detector acts on internal model features.

        Returns:
            bool: True if the detector perform computations on an intermediate layer
            else False.
        """
        return False
