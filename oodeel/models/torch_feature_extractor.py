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

import numpy as np
import torch
from torch import nn

FEATURES = dict()


def get_features_hook(key: str, store_inputs: bool = False):
    """
    Hook that stores the tensor value into a global dictionnary with a given key so that it can be easily retrieve.
    """
    global FEATURES
    if not (key in FEATURES.keys()):
        FEATURES[key] = list()

    def hook(m, i, o):
        if store_inputs:
            tens = i
        else:
            tens = o
        if isinstance(tens, torch.Tensor):
            FEATURES[key].append([tens.detach()])
        elif isinstance(tens, (list, tuple)):
            FEATURES[key].append([_.detach() for _ in tens])
        else:
            print(tens)
            print(type(tens))
            raise NotImplementedError

    return hook


class TorchFeatureExtractor(object):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers, or the output of the model (softmax/logits).

    Args:
        model: model to extract the features from
        output_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer. Defaults to [].
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of the model).
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
        output_activation: str = None,
        flatten: bool = False,
        batch_size: int = None,
    ):

        self.model = model
        self.output_layers_id = output_layers_id
        self.input_layer_id = input_layer_id
        self.output_activation = output_activation
        self.flatten = flatten
        self.batch_size = batch_size

        # Register a hook to store feature values for each considered layer.
        _layer_store = dict()
        for layer_name, layer in self.model.named_modules():
            if layer_name in self.output_layers_id:
                _layer_store[layer_name] = layer

        for layer_id in self.output_layers_id:
            if isinstance(layer_id, str):
                _layer_store[layer_id].register_forward_hook(
                    get_features_hook(layer_id)
                )
            elif isinstance(layer_id, int):
                self.model[layer_id].register_forward_hook(get_features_hook(layer_id))
            else:
                raise NotImplementedError

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

    def predict_on_batch(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get features associated with the input. Works on an in-memory tensor.
        """
        global FEATURES
        # Empty feature dict
        for key in FEATURES.keys():
            FEATURES[key] = list()

        model_outputs = self.model(x)

        flatten_fn = (
            (lambda t: torch.flatten(t, start_dim=1)) if self.flatten else (lambda t: t)
        )
        output_act_fn = (
            getattr(nn, self.output_activation)()
            if self.output_activation
            else (lambda t: t)
        )

        def process_outputs_fn(t: torch.Tensor) -> torch.Tensor:
            return output_act_fn(flatten_fn(t))

        return [
            process_outputs_fn(FEATURES[layer_id][0][0])
            for layer_id in self.output_layers_id
        ]

    def predict(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features for a given inputs. If batch_size is specified, the model is called on batches and outputs are concatenated.
        """
        if self.batch_size:
            batch_bounds = np.arange(0, x.size(0), self.batch_size).tolist() + [
                x.size(0)
            ]
            _batch_results = list()
            for low, up in zip(batch_bounds[:-1], batch_bounds[1:]):
                _batch = x[low:up, :, :, :]
                _batch_results.append(self.predict_on_batch(_batch))

            n_features = len(_batch_results[0])

            _features = list()

            for feature_idx in range(n_features):
                _features.append(
                    torch.cat([_batch[feature_idx] for _batch in _batch_results])
                )

            return _features

        else:
            return self.predict_on_batch(x)
