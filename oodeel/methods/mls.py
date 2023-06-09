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
from ..types import Optional
from ..types import TensorType
from ..types import Union
from .base import OODBaseDetector


class MLS(OODBaseDetector):
    """
    Maximum Logit Scores method for OOD detection.
    "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?"
    https://arxiv.org/abs/2110.06207,
    and Maximum Softmax Score
    "A Baseline for Detecting Misclassified and Out-of-Distribution Examples
    in Neural Networks"
    http://arxiv.org/abs/1610.02136

    Args:
        output_activation (str): activation function for the last layer. If "linear",
            the method is MLS and if "softmax", the method is MSS.
            Defaults to "linear".
        react_quantile: if not None, a threshold corresponding to this quantile for the
            penultimate layer activations is calculated, then used to clip the
            activations under this threshold (ReAct method). Defaults to None.
        penultimate_layer_id: identifier for the penultimate layer, used for ReAct.
            Defaults to None.
    """

    def __init__(
        self,
        output_activation: str = "linear",
        react_quantile: Optional[float] = None,
        penultimate_layer_id: Optional[Union[str, int]] = None,
    ):
        super().__init__(
            output_layers_id=[-1],
            react_quantile=react_quantile,
            penultimate_layer_id=penultimate_layer_id,
        )
        self.output_activation = output_activation

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """

        pred = self.feature_extractor(inputs)
        if self.output_activation == "softmax":
            pred = self.op.softmax(pred)
        pred = self.op.convert_to_numpy(pred)
        scores = -np.max(pred, axis=1)
        return scores

    def _fit_to_dataset(self, fit_dataset: DatasetType) -> None:
        """
        Fits the OOD detector to fit_dataset.

        Args:
            fit_dataset: dataset to fit the OOD detector on
        """
        pass
