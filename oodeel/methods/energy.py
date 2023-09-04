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
from scipy.special import logsumexp

from ..types import DatasetType
from ..types import TensorType
from ..types import Tuple
from .base import OODBaseDetector


class Energy(OODBaseDetector):
    r"""
    Energy Score method for OOD detection.
    "Energy-based Out-of-distribution Detection"
    https://arxiv.org/abs/2010.03759

    This method assumes that the model has been trained with cross entropy loss
    $CE(model(x))$ where $model(x)=(l_{c})_{c=1}^{C}$ are the logits
    predicted for input $x$.
    The implementation assumes that the logits are retreieved using the output with
    linear activation.

    The energy score for input $x$ is given by
    $$ -\log \sum_{c=0}^C \exp(l_c)$$

    where $model(x)=(l_{c})_{c=1}^{C}$ are the logits predicted by the model on
    $x$.
    As always, training data is expected to have lower score than OOD data.

    Args:
        use_react (bool): if true, apply ReAct method by clipping penultimate
            activations under a threshold value.
        react_quantile (Optional[float]): q value in the range [0, 1] used to compute
            the react clipping threshold defined as the q-th quantile penultimate layer
            activations. Defaults to 0.8.
    """

    def __init__(
        self,
        use_react: bool = False,
        react_quantile: float = 0.8,
    ):
        super().__init__(
            use_react=use_react,
            react_quantile=react_quantile,
        )

    def _score_tensor(self, inputs: TensorType) -> Tuple[np.ndarray]:
        """
        Computes an OOD score for input samples "inputs" based on
        energy, namey $-logsumexp(logits(inputs))$.

        Args:
            inputs: input samples to score

        Returns:
            Tuple[np.ndarray]: scores, logits
        """
        # compute logits (softmax(logits,axis=1) is the actual softmax
        # output minimized using binary cross entropy)
        _, logits = self.feature_extractor.predict_tensor(inputs)
        logits = self.op.convert_to_numpy(logits)
        scores = -logsumexp(logits, axis=1)
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
