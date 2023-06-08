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
from collections import OrderedDict
from typing import get_args
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.torch_data_handler import TorchDataHandler
from ..types import Callable
from ..types import ItemType
from ..types import List
from ..types import TensorType
from ..types import Union
from ..utils.torch_operator import sanitize_input
from .feature_extractor import FeatureExtractor


class TorchFeatureExtractor(FeatureExtractor):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers,
    or the output of the model (softmax/logits).

    Args:
        model: model to extract the features from
        output_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer. Defaults to [].
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of
            the model).
            Defaults to None.
        react_threshold: if not None, penultimate layer activations are clipped under
            this threshold value (useful for ReAct). Defaults to None.
        penultimate_layer_id: identifier for the penultimate layer, used for ReAct.
            Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        output_layers_id: List[Union[int, str]] = [],
        input_layer_id: Optional[Union[int, str]] = None,
        react_threshold: Optional[float] = None,
        penultimate_layer_id: Optional[Union[str, int]] = None,
    ):
        model = model.eval()
        super().__init__(
            model=model,
            output_layers_id=output_layers_id,
            input_layer_id=input_layer_id,
            react_threshold=react_threshold,
            penultimate_layer_id=penultimate_layer_id,
        )
        self._device = next(model.parameters()).device
        self._features = {layer: torch.empty(0) for layer in self.output_layers_id}
        self.backend = "torch"

    def _get_features_hook(self, layer_id: Union[str, int]) -> Callable:
        """
        Hook that stores features corresponding to a specific layer
        in a class dictionary.

        Args:
            layer_id (Union[str, int]): layer identifier

        Returns:
            Callable: hook function
        """
        self._features = {layer: torch.empty(0) for layer in self.output_layers_id}

        def hook(_, __, output):
            if isinstance(output, torch.Tensor):
                self._features[layer_id] = output
            else:
                raise NotImplementedError

        return hook

    def find_layer(self, layer_id: Union[str, int]) -> nn.Module:
        """Find a layer in a model either by his name or by his index.

        Args:
            layer_id (Union[str, int]): layer identifier

        Returns:
            nn.Module: the corresponding layer
        """
        if isinstance(layer_id, int):
            if isinstance(self.model, nn.Sequential):
                return self.model[layer_id]
            else:
                return list(self.model.named_modules())[layer_id][1]
        else:
            return dict(self.model.named_modules())[layer_id]

    def prepare_extractor(self) -> None:
        """Prepare the feature extractor by adding hooks to self.model"""
        # remove forward hooks attached to the model
        self._clean_forward_hooks()

        # === If react method, clip activations from penultimate layer ===
        if self.react_threshold is not None:
            assert isinstance(self.react_threshold, (float, int))
            assert self.penultimate_layer_id is not None
            pen_layer = self.find_layer(self.penultimate_layer_id)
            pen_layer.register_forward_hook(self._get_clip_hook(self.react_threshold))

        # Register a hook to store feature values for each considered layer.
        for layer_id in self.output_layers_id:
            layer = self.find_layer(layer_id)
            layer.register_forward_hook(self._get_features_hook(layer_id))

        # Crop model if input layer is provided
        if not (self.input_layer_id) is None:
            if isinstance(self.input_layer_id, int):
                if isinstance(self.model, nn.Sequential):
                    self.model = nn.Sequential(
                        *list(self.model.modules())[self.input_layer_id :]
                    )
                else:
                    raise NotImplementedError
            elif isinstance(self.input_layer_id, str):
                if isinstance(self.model, nn.Sequential):
                    module_names = list(
                        filter(
                            lambda x: x != "",
                            map(lambda x: x[0], self.model.named_modules()),
                        )
                    )
                    input_module_idx = module_names.index(self.input_layer_id)
                    self.model = nn.Sequential(
                        *list(self.model.modules())[(input_module_idx + 1) :]
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    @sanitize_input
    def predict_tensor(
        self, x: TensorType, detach: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the projection of tensor in the feature space of self.model

        Args:
            x (TensorType): input tensor (or dataset elem)
            detach (bool): if True, return features detached from the computational graph.
                Defaults to True.

        Returns:
            List[torch.Tensor]: features
        """
        if x.device != self._device:
            x = x.to(self._device)
        _ = self.model(x)

        if detach:
            features = [
                self._features[layer_id].detach() for layer_id in self.output_layers_id
            ]
        else:
            features = [self._features[layer_id] for layer_id in self.output_layers_id]

        if len(features) == 1:
            features = features[0]
        return features

    def predict(
        self, dataset: Union[DataLoader, ItemType], detach: bool = True, **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (Union[DataLoader, ItemType]): input dataset
            detach (bool): if True, return features detached from the computational graph.
                Defaults to True.
            kwargs (dict): additional arguments not considered for prediction

        Returns:
            List[torch.Tensor]: features
        """

        if isinstance(dataset, get_args(ItemType)):
            tensor = TorchDataHandler.get_input_from_dataset_item(dataset)
            return self.predict_tensor(tensor, detach=detach)

        features = [None for i in range(len(self.output_layers_id))]
        for elem in tqdm(
            dataset, desc="Extracting the dataset features...", total=len(dataset)
        ):
            tensor = TorchDataHandler.get_input_from_dataset_item(elem)
            features_batch = self.predict_tensor(tensor, detach=detach)
            if len(features) == 1:
                features_batch = [features_batch]
            for i, f in enumerate(features_batch):
                features[i] = (
                    f if features[i] is None else torch.cat([features[i], f], dim=0)
                )

        # No need to return a list when there is only one input layer
        if len(features) == 1:
            features = features[0]
        return features

    def get_weights(self, layer_id: Union[str, int]) -> List[torch.Tensor]:
        """Get the weights of a layer

        Args:
            layer_id (Union[int, str]): layer identifier

        Returns:
            List[torch.Tensor]: weights and biases matrixes
        """
        layer = self.find_layer(layer_id)
        return [layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()]

    def _get_clip_hook(self, threshold: float) -> Callable:
        """
        Hook that truncate activation features under a threshold value

        Args:
            threshold (float): threshold value

        Returns:
            Callable: hook function
        """

        def hook(_, __, output):
            output = torch.clip(output, max=threshold)
            return output

        return hook

    def _clean_forward_hooks(self) -> None:
        """
        Remove all the forward hook attached to the model's layers. This function should
        be called at the __init__, and avoid to accumulate the hooks when defining a
        new TorchFeatureExtractor for the same model.
        """

        def __clean_hooks(m: nn.Module):
            for _, child in m._modules.items():
                if child is not None:
                    if hasattr(child, "_forward_hooks"):
                        child._forward_hooks = OrderedDict()
                    __clean_hooks(child)

        return __clean_hooks(self.model)
