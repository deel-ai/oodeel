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


class ODIN(OODBaseDetector):
    """ "Enhancing The Reliability of Out-of-distribution Image Detection
    in Neural Networks"
    http://arxiv.org/abs/1706.02690

    Args:
        temperature (float, optional): Temperature parameter. Defaults to 1000.
        eps (float, optional): Perturbation noise. Defaults to 0.014.
        use_react (bool): if true, apply ReAct method by clipping penultimate
            activations under a threshold value.
        react_quantile (Optional[float]): q value in the range [0, 1] used to compute
            the react clipping threshold defined as the q-th quantile penultimate layer
            activations. Defaults to 0.8.
    """

    def __init__(
        self,
        temperature: float = 1000,
        eps: float = 0.014,
        use_react: bool = False,
        use_scale: bool = False,
        use_ash: bool = False,
        react_quantile: float = 0.8,
        scale_percentile: float = 0.85,
        ash_percentile: float = 0.90,
        **kwargs,
    ):
        super().__init__(
            eps=eps,
            temperature=temperature,
            use_react=use_react,
            use_scale=use_scale,
            use_ash=use_ash,
            react_quantile=react_quantile,
            scale_percentile=scale_percentile,
            ash_percentile=ash_percentile,
            **kwargs,
        )

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs (TensorType): input samples to score

        Returns:
            Tuple[np.ndarray]: scores, logits
        """
        if self.eps > 0:
            x = self._input_perturbation(inputs, self.eps, self.temperature)

        _, logits = self.feature_extractor.predict_tensor(x)
        logits_s = logits / self.temperature
        probits = self.op.softmax(logits_s)
        probits = self.op.convert_to_numpy(probits)
        scores = -np.max(probits, axis=1)
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
