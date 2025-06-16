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
from typing import Optional

import torch
from torch import nn

from ..types import Callable
from ..types import List
from ..types import TensorType
from ..types import Tuple
from ..types import Union
from ..utils.torch_operator import sanitize_input
from .torch_feature_extractor import TorchFeatureExtractor


class HFTorchFeatureExtractor(TorchFeatureExtractor):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers,
    or the output of the model (logits).

    Args:
        model: model to extract the features from
        feature_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Important: for HFTorchFeatureExtractor, we use features from the
            hidden states returned by model(input, output_hidden_states=True) in
            addition to other features computed like in TorchFeatureExtractor.
            To select the hidden states as feature, identify the layer by hidden_i,
            with i the index of the hidden state.
            Defaults to [].
        head_layer_id (int, str): identifier of the head layer.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            We recommend to keep the default value for HFTorchFeatureExtractor unless
            you know what you are doing.
            Defaults to -1
        react_threshold: if not None, penultimate layer activations are clipped under
            this threshold value (useful for ReAct). Defaults to None.
        scale_percentile: if not None, the features are scaled
            following the method of Xu et al., ICLR 2024.
            Defaults to None.
        ash_percentile: if not None, the features are scaled following
            the method of Djurisic et al., ICLR 2023.
        return_penultimate (bool): if True, the penultimate values are returned,
            i.e. the input to the head_layer.

    """

    def __init__(
        self,
        model: nn.Module,
        feature_layers_id: List[int] = [],
        head_layer_id: Optional[Union[int, str]] = -1,
        react_threshold: Optional[float] = None,
        scale_percentile: Optional[float] = None,
        ash_percentile: Optional[float] = None,
        return_penultimate: Optional[bool] = False,
    ):
        super().__init__(
            model=model,
            feature_layers_id=feature_layers_id,
            head_layer_id=head_layer_id,
            react_threshold=react_threshold,
            scale_percentile=scale_percentile,
            ash_percentile=ash_percentile,
            return_penultimate=return_penultimate,
        )

        self._features = {layer: torch.empty(0) for layer in self._hook_layers_id}

    def _parse_hf_hidden_state(self, feature_layer_id) -> Tuple[bool, Union[int, str]]:
        """Parse the feature_layer_id to check if it is a hidden state from HF model.
        If it is, return True and the index of the hidden state.
        If it is not, return False and the feature_layer_id.

        Args:
            feature_layer_id (Union[int, str]): feature layer id to parse.

        Returns:
            Tuple[bool, Union[int, str]]: is_hf_hidden_state, feature_layer_id
        """
        if (
            isinstance(feature_layer_id, str)
            and len(feature_layer_id) >= 7
            and feature_layer_id[:7] == "hidden_"
        ):
            return True, int(feature_layer_id[7:])

        return False, feature_layer_id

    @property
    def _hook_layers_id(self) -> List[Union[int, str]]:
        """Get the list of hook layer ids to be used for feature extraction.
        This list excludes hf_hidden_states because it feature extraction is already
        handled by HF transformers in that case.
        """
        hook_layer_ids = []
        for feature_layer_id in self.feature_layers_id:
            if (
                isinstance(feature_layer_id, str)
                and len(feature_layer_id) >= 7
                and feature_layer_id[:7] == "hidden_"
            ):
                continue

            hook_layer_ids.append(feature_layer_id)

        return hook_layer_ids

    @sanitize_input
    def predict_tensor(
        self,
        x: TensorType,
        postproc_fns: Optional[List[Callable]] = None,
        detach: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Get the projection of tensor in the feature space of self.model

        Args:
            x (TensorType): input tensor (or dataset elem)
            postproc_fns (Optional[List[Callable]]): postprocessing function to apply to
                each feature immediately after forward. Default to None.
            detach (bool): if True, return features detached from the computational
                graph. Defaults to True.

        Returns:
            List[torch.Tensor], torch.Tensor: features, logits
        """
        if x.device != self._device:
            x = x.to(self._device)
        outputs = self.model(x, output_hidden_states=True, return_dict=True)

        features = []
        for feature_layer_id in self.feature_layers_id:
            is_hf_hidden_state, feature_layer_id = self._parse_hf_hidden_state(
                feature_layer_id
            )
            if is_hf_hidden_state:
                features.append(
                    outputs["hidden_states"][feature_layer_id].detach()
                    if detach
                    else outputs["hidden_states"][feature_layer_id]
                )
            else:
                features.append(
                    self._features[feature_layer_id].detach()
                    if detach
                    else self._features[feature_layer_id]
                )

        logits = outputs.logits.detach() if detach else outputs.logits

        if postproc_fns is not None:
            features = [
                postproc_fn(feature)
                for feature, postproc_fn in zip(features, postproc_fns)
            ]

        self._last_logits = logits
        return features, logits
