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
from ..types import Tuple
from ..types import Union
from ..utils.torch_operator import sanitize_input
from .feature_extractor import FeatureExtractor


class TorchFeatureExtractor(FeatureExtractor):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers,
    or the output of the model (logits).

    Args:
        model: model to extract the features from
        feature_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer. Defaults to [].
        head_layer_id (int, str): identifier of the head layer.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Defaults to -1
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of
            the model).
            Defaults to None.
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
        feature_layers_id: List[Union[int, str]] = [],
        head_layer_id: Optional[Union[int, str]] = -1,
        input_layer_id: Optional[Union[int, str]] = None,
        react_threshold: Optional[float] = None,
        scale_percentile: Optional[float] = None,
        ash_percentile: Optional[float] = None,
        return_penultimate: Optional[bool] = False,
    ):
        model = model.eval()

        if return_penultimate:
            feature_layers_id.append("penultimate")

        super().__init__(
            model=model,
            feature_layers_id=feature_layers_id,
            head_layer_id=head_layer_id,
            input_layer_id=input_layer_id,
            react_threshold=react_threshold,
            scale_percentile=scale_percentile,
            ash_percentile=ash_percentile,
            return_penultimate=return_penultimate,
        )
        self._device = next(model.parameters()).device
        self._features = {layer: torch.empty(0) for layer in self._hook_layers_id}
        self._last_logits = None
        self.backend = "torch"

    @property
    def _hook_layers_id(self):
        return self.feature_layers_id + [self.head_layer_id]

    def _get_features_hook(self, layer_id: Union[str, int]) -> Callable:
        """
        Hook that stores features corresponding to a specific layer
        in a class dictionary.

        Args:
            layer_id (Union[str, int]): layer identifier

        Returns:
            Callable: hook function
        """

        def hook(_, __, output):
            if isinstance(output, torch.Tensor):
                self._features[layer_id] = output
            else:
                raise NotImplementedError

        return hook

    def _get_penultimate_hook(self) -> Callable:
        """
        Hook that stores features corresponding to a specific layer
        in a class dictionary.

        Args:
            layer_id (Union[str, int]): layer identifier

        Returns:
            Callable: hook function
        """

        def hook(_, input):
            if isinstance(input[0], torch.Tensor):
                self._features["penultimate"] = input[0]
            else:
                raise NotImplementedError

        return hook

    @staticmethod
    def find_layer(
        model: nn.Module,
        layer_id: Union[str, int],
        index_offset: int = 0,
        return_id: bool = False,
    ) -> Union[nn.Module, Tuple[nn.Module, str]]:
        """Find a layer in a model either by his name or by his index.

        Args:
            model (nn.Module): model whose identified layer will be returned
            layer_id (Union[str, int]): layer identifier
            index_offset (int): index offset to find layers located before (negative
                offset) or after (positive offset) the identified layer
            return_id (bool): if True, the layer will be returned with its id

        Returns:
            Union[nn.Module, Tuple[nn.Module, str]]: the corresponding layer and its id
                if return_id is True.
        """
        if isinstance(layer_id, int):
            layer_id += index_offset
            if isinstance(model, nn.Sequential):
                layer = model[layer_id]
            else:
                layer = list(model.named_modules())[layer_id][1]
        else:
            layer_id = list(dict(model.named_modules()).keys()).index(layer_id)
            layer_id += index_offset
            layer = list(model.named_modules())[layer_id][1]

        if return_id:
            return layer, layer_id
        else:
            return layer

    def prepare_extractor(self) -> None:
        """Prepare the feature extractor by adding hooks to self.model"""
        # prepare self.model for ood hooks (add _ood_handles attribute or
        # remove ood forward hooks attached to the model)
        self._prepare_ood_handles()

        # === If react method, clip activations from penultimate layer ===
        if self.react_threshold is not None:
            pen_layer = self.find_layer(self.model, self.head_layer_id)
            self.model._ood_handles.append(
                pen_layer.register_forward_pre_hook(
                    self._get_clip_hook(self.react_threshold)
                )
            )

        # === If SCALE method, scale activations from penultimate layer ===
        if self.scale_percentile is not None:
            pen_layer = self.find_layer(self.model, self.head_layer_id)
            self.model._ood_handles.append(
                pen_layer.register_forward_pre_hook(
                    self._get_scale_hook(self.scale_percentile)
                )
            )

        # === If ASH method, scale and prune activations from penultimate layer ===
        if self.ash_percentile is not None:
            pen_layer = self.find_layer(self.model, self.head_layer_id)
            self.model._ood_handles.append(
                pen_layer.register_forward_pre_hook(
                    self._get_ash_hook(self.ash_percentile)
                )
            )

        # Register a hook to store feature values for each considered layer + last layer
        for layer_id in self._hook_layers_id:
            if layer_id == "penultimate":
                # Register penultimate hook
                layer = self.find_layer(self.model, self.head_layer_id)
                self.model._ood_handles.append(
                    layer.register_forward_pre_hook(self._get_penultimate_hook())
                )
                continue

            layer = self.find_layer(self.model, layer_id)
            self.model._ood_handles.append(
                layer.register_forward_hook(self._get_features_hook(layer_id))
            )

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
        _ = self.model(x)

        if detach:
            features = [
                self._features[layer_id].detach() for layer_id in self._hook_layers_id
            ]
        else:
            features = [self._features[layer_id] for layer_id in self._hook_layers_id]

        # split features and logits
        logits = features.pop()

        if postproc_fns is not None:
            features = [
                postproc_fn(feature)
                for feature, postproc_fn in zip(features, postproc_fns)
            ]

        self._last_logits = logits
        return features, logits

    def predict(
        self,
        dataset: Union[DataLoader, ItemType],
        postproc_fns: Optional[List[Callable]] = None,
        detach: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], dict]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (Union[DataLoader, ItemType]): input dataset
            postproc_fns (Optional[List[Callable]]): postprocessing function to apply to
                each feature immediately after forward. Default to None.
            detach (bool): if True, return features detached from the computational
                graph. Defaults to True.
            verbose (bool): if True, display a progress bar. Defaults to False.
            kwargs (dict): additional arguments not considered for prediction

        Returns:
            List[torch.Tensor], dict: features and extra information (logits, labels) as
                a dictionary.
        """
        labels = None

        if isinstance(dataset, get_args(ItemType)):
            tensor = TorchDataHandler.get_input_from_dataset_item(dataset)
            features, logits = self.predict_tensor(tensor, postproc_fns, detach=detach)

            # Get labels if dataset is a tuple/list
            if isinstance(dataset, (list, tuple)) and len(dataset) > 1:
                labels = TorchDataHandler.get_label_from_dataset_item(dataset)

        else:
            features = [None for i in range(len(self.feature_layers_id))]
            logits = None
            batch = next(iter(dataset))
            contains_labels = isinstance(batch, (list, tuple)) and len(batch) > 1
            for elem in tqdm(dataset, desc="Predicting", disable=not verbose):
                tensor = TorchDataHandler.get_input_from_dataset_item(elem)
                features_batch, logits_batch = self.predict_tensor(
                    tensor, postproc_fns, detach=detach
                )
                for i, f in enumerate(features_batch):
                    features[i] = (
                        f if features[i] is None else torch.cat([features[i], f], dim=0)
                    )
                # concatenate logits
                logits = (
                    logits_batch
                    if logits is None
                    else torch.cat([logits, logits_batch], axis=0)
                )
                # concatenate labels of current batch with previous batches
                if contains_labels:
                    lbl_batch = TorchDataHandler.get_label_from_dataset_item(elem)

                    if labels is None:
                        labels = lbl_batch
                    else:
                        labels = torch.cat([labels, lbl_batch], dim=0)

        # store extra information in a dict
        info = dict(labels=labels, logits=logits)
        return features, info

    def get_weights(self, layer_id: Union[str, int]) -> List[torch.Tensor]:
        """Get the weights of a layer

        Args:
            layer_id (Union[int, str]): layer identifier

        Returns:
            List[torch.Tensor]: weights and biases matrixes
        """
        layer = self.find_layer(self.model, layer_id)
        return [layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()]

    def _get_clip_hook(self, threshold: float) -> Callable:
        """
        Hook that truncate activation features under a threshold value

        Args:
            threshold (float): threshold value

        Returns:
            Callable: hook function
        """

        def hook(_, input):
            input = input[0]
            input = torch.clip(input, max=threshold)
            return input

        return hook

    def _get_scale_hook(self, percentile: float) -> Callable:
        """
        Hook that scales activation features.

        Args:
            threshold (float): threshold value

        Returns:
            Callable: hook function
        """

        def hook(_, input):
            input = input[0]
            output_percentile = torch.quantile(input, percentile, dim=1)
            mask = input > output_percentile[:, None]
            output_masked = input * mask
            s = torch.exp(torch.sum(input, dim=1) / torch.sum(output_masked, dim=1))
            s = torch.unsqueeze(s, 1)
            input = input * s
            return input

        return hook

    def _get_ash_hook(self, percentile: float) -> Callable:
        """
        Hook that scales and prunes activation features under a threshold value

        Args:
            threshold (float): threshold value

        Returns:
            Callable: hook function
        """

        def hook(_, input):
            input = input[0]
            output_percentile = torch.quantile(input, percentile, dim=1)
            mask = input > output_percentile[:, None]
            output_masked = input * mask
            s = torch.exp(torch.sum(input, dim=1) / torch.sum(output_masked, dim=1))
            s = torch.unsqueeze(s, 1)
            input = output_masked * s
            return input

        return hook

    def _prepare_ood_handles(self) -> None:
        """
        Prepare the model by either setting a new attribute to self.model
        as a list which will contain all the ood specific hooks, or by cleaning
        the existing ood specific hooks if the attribute already exists.
        """

        if not hasattr(self.model, "_ood_handles"):
            setattr(self.model, "_ood_handles", [])
        else:
            for handle in self.model._ood_handles:
                handle.remove()
            self.model._ood_handles = []
