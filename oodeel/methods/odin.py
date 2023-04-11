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

from ..types import TensorType
from .base import OODModel


class ODIN(OODModel):
    """ "Enhancing The Reliability of Out-of-distribution Image Detection
    in Neural Networks"
    http://arxiv.org/abs/1706.02690

    Args:
        temperature (float, optional): Temperature parameter. Defaults to 1000.
        noise (float, optional): Perturbation noise. Defaults to 0.014.
    """

    def __init__(
        self,
        temperature: float = 1000,
        noise: float = 0.014,
    ):
        self.temperature = temperature
        super().__init__(output_layers_id=[-1])
        self.noise = noise

    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        if self.feature_extractor.backend == "torch":
            inputs = inputs.to(self.feature_extractor._device)
        x = self.input_perturbation(inputs)
        logits = self.feature_extractor.model(x) / self.temperature
        pred = self.op.softmax(logits)
        pred = self.op.convert_to_numpy(pred)
        scores = -np.max(pred, axis=1)
        return scores

    def input_perturbation(self, inputs):
        preds = self.feature_extractor.model(inputs)
        outputs = self.op.argmax(preds, dim=1)
        gradients = self.op.gradient(self._temperature_loss, inputs, outputs)
        inputs_p = inputs - self.noise * self.op.sign(gradients)
        return inputs_p

    def _temperature_loss(self, inputs, labels):
        preds = self.feature_extractor.model(inputs) / self.temperature
        loss = self.op.CrossEntropyLoss(reduction="sum")(inputs=preds, targets=labels)
        return loss
