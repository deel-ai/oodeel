from dataclasses import dataclass, field
from typing import List, Union
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
            FEATURES[key].append([tens.detach().numpy()])
        elif isinstance(tens, (list, tuple)):
            FEATURES[key].append([_.detach().numpy() for _ in tens])
        else:
            print(tens)
            print(type(tens))
            raise NotImplementedError

    return hook


@dataclass
class TorchFeatureExtractor(object):
    model: nn.Module
    output_layers_id: List[Union[int, str]] = field(default_factory=list)
    input_layer_id: Union[int, str] = None
    output_activation: str = None
    flatten: bool = False
    batch_size: int = None

    def __post_init__(self):
        _layer_store = dict()
        for layer_name, layer in self.model.named_modules():
            if layer_name in self.output_layers_id:
                _layer_store[layer_name] = layer

        for layer_id in self.output_layers_id:
            if isinstance(layer_id, str):
                _layer_store[layer_id].register_forward_hook(get_features_hook(layer_id))
            elif isinstance(layer_id, int):
                self.model[layer_id].register_forward_hook(get_features_hook(layer_id))
            else:
                raise NotImplementedError

    def predict_on_batch(self, x: torch.Tensor) -> List[torch.Tensor]:
        global FEATURES
        # Empty feature dict
        for key in FEATURES.keys():
            FEATURES[key] = list()

        model_outputs = self.model(x)

        flatten_fn = (lambda t: torch.flatten(t, 1)) if self.flatten else (lambda t: t)
        output_act_fn = getattr(nn, self.output_activation)() if self.output_activation else (lambda t: t)

        def process_outputs_fn(t: torch.Tensor) -> torch.Tensor:
            return output_act_fn(flatten_fn(t))

        return [process_outputs_fn(FEATURES[layer_id][0][0]) for layer_id in self.output_layers_id]

    def predict(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.batch_size:
            batch_bounds = np.arange(0, x.size(0), self.batch_size).tolist() + [x.size(0)]
            _batch_results = list()
            for low, up in zip(batch_bounds[:-1], batch_bounds[1:]):
                _batch = x[low:up, :, :, :]
                _batch_results.append(self.predict_on_batch(_batch))

            n_features = len(_batch_results[0])

            _features = list()

            for feature_idx in range(n_features):
                _features.append(np.vstack([_batch[feature_idx] for _batch in _batch_results]))

            return _features

        else:
            return self.predict_on_batch(x)
