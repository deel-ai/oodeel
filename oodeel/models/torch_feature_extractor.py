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
from typing import List
from typing import Union

import torch
from torch import nn

from ..utils.tf_tools import get_input_from_dataset_elem
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
        output_activation:  activation function for the last layer.
            Defaults to None.
        flatten: Flatten the output features or not.
            Defaults to True.
        batch_size: batch_size used to compute the features space
            projection of input data.
            Defaults to 256.
    """

    def __init__(
        self,
        model: nn.Module,
        output_layers_id: List[Union[int, str]] = [],
        input_layer_id: Union[int, str] = None,
    ):
        super().__init__(
            model=model,
            output_layers_id=output_layers_id,
            input_layer_id=input_layer_id,
        )
        self._device = next(model.parameters()).device
        self._features = {layer: torch.empty(0) for layer in output_layers_id}

    def get_features_hook(self, layer_id: Union[str, int]):
        """
        Hook that stores features corresponding to a specific layer
        in a class dictionary.
        """

        def hook(_, __, output):
            if isinstance(output, torch.Tensor):
                self._features[layer_id] = output
            else:
                raise NotImplementedError

        return hook

    def find_layer(self, layer_id: Union[str, int]) -> nn.Module:
        """
        Find a layer in a model either by his name or by his index.
        Parameters
        ----------
        model
            Model on which to search.
        layer
            Layer name or layer index
        Returns
        -------
        layer
            Layer found
        """
        if isinstance(layer_id, int):
            assert isinstance(self.model, nn.Sequential), (
                "The model must be "
                "subscriptable to find output layer by int identifier"
            )
            return self.model[layer_id]
        else:
            for layer_name, layer in self.model.named_modules():
                if layer_name == layer_id:
                    return layer

    def prepare_extractor(self):
        """
        Prepare the feature extractor for inference.
        """
        # Register a hook to store feature values for each considered layer.
        for layer_id in self.output_layers_id:
            layer = self.find_layer(layer_id)
            layer.register_forward_hook(self.get_features_hook(layer_id))

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

    def predict_tensor(self, x: torch.Tensor, detach=False) -> List[torch.Tensor]:
        """
        Get features associated with the input. Works on an in-memory tensor.
        """

        if x.device != self._device:
            x = x.to(self._device)
        _ = self.model(x)

        features = [
            self._features[layer_id]
            if not detach
            else self._features[layer_id].detach()
            for layer_id in self.output_layers_id
        ]

        if len(features) == 1:
            features = features[0]
        return features

    def predict(
        self, dataset: torch.utils.data.DataLoader, detach=False
    ) -> List[torch.Tensor]:
        """
        Extract features for a given inputs. If batch_size is specified,
        the model is called on batches and outputs are concatenated.
        """

        if not isinstance(dataset, torch.utils.data.DataLoader):
            tensor = get_input_from_dataset_elem(dataset)
            return self.predict_tensor(tensor, detach)

        features = [None for i in range(len(self.output_layers_id))]
        for elem in dataset:
            tensor = get_input_from_dataset_elem(elem)
            features_batch = self.predict_tensor(tensor, detach)
            if len(features) == 1:
                features_batch = [features_batch]
            for i, f in enumerate(features_batch):
                features[i] = (
                    f if features[i] is None else torch.concat([features[i], f], dim=0)
                )

        # No need to return a list when there is only one input layer
        if len(features) == 1:
            features = features[0]
        return features

    def get_weights(self, layer_id: Union[str, int]):
        """
        Constructs the feature extractor model

        Returns:
        """
        layer = self.find_layer(layer_id)
        return [layer.weight.detach().numpy(), layer.bias.detach().numpy()]
