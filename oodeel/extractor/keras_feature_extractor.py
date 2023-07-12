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

import tensorflow as tf

from ..datasets.tf_data_handler import TFDataHandler
from ..types import Callable
from ..types import ItemType
from ..types import List
from ..types import TensorType
from ..types import Tuple
from ..types import Union
from ..utils.tf_operator import sanitize_input
from .feature_extractor import FeatureExtractor


class KerasFeatureExtractor(FeatureExtractor):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers, or the output of the model
    (softmax/logits).

    Args:
        model: model to extract the features from
        output_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer. Defaults to [].
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of the
            model).
            Defaults to None.
        react_threshold: if not None, penultimate layer activations are clipped under
            this threshold value (useful for ReAct). Defaults to None.
    """

    def __init__(
        self,
        model: Callable,
        output_layers_id: List[Union[int, str]] = [-1],
        input_layer_id: Optional[Union[int, str]] = None,
        react_threshold: Optional[float] = None,
    ):
        if input_layer_id is None:
            input_layer_id = 0
        super().__init__(
            model=model,
            output_layers_id=output_layers_id,
            input_layer_id=input_layer_id,
            react_threshold=react_threshold,
        )

        self.backend = "tensorflow"
        self.model.layers[-1].activation = getattr(tf.keras.activations, "linear")

    @staticmethod
    def find_layer(
        model: Callable,
        layer_id: Union[str, int],
        index_offset: int = 0,
        return_id: bool = False,
    ) -> Union[tf.keras.layers.Layer, Tuple[tf.keras.layers.Layer, str]]:
        """Find a layer in a model either by his name or by his index.

        Args:
            model (Callable): model whose identified layer will be returned
            layer_id (Union[str, int]): layer identifier
            index_offset (int): index offset to find layers located before (negative
                offset) or after (positive offset) the identified layer
            return_id (bool): if True, the layer will be returned with its id

        Raises:
            ValueError: if the layer is not found

        Returns:
            Union[tf.keras.layers.Layer, Tuple[tf.keras.layers.Layer, str]]:
                the corresponding layer and its id if return_id is True.
        """
        if isinstance(layer_id, str):
            layers_names = [layer.name for layer in model.layers]
            layer_id = layers_names.index(layer_id)
        if isinstance(layer_id, int):
            layer_id += index_offset
            layer = model.get_layer(index=layer_id)
        else:
            raise ValueError(f"Could not find any layer {layer_id}.")

        if return_id:
            return layer, layer_id
        else:
            return layer

    # @tf.function
    # TODO check with Thomas about @tf.function
    def prepare_extractor(self) -> tf.keras.models.Model:
        """Constructs the feature extractor model

        Returns:
            tf.keras.models.Model: truncated model (extractor)
        """
        input_layer = self.find_layer(self.model, self.input_layer_id)
        new_input = tf.keras.layers.Input(tensor=input_layer.input)
        output_tensors = [
            self.find_layer(self.model, ol_id).output for ol_id in self.output_layers_id
        ]

        # === If react method, clip activations from penultimate layer ===
        if self.react_threshold is not None:
            penultimate_layer, penultimate_layer_id = self.find_layer(
                self.model, self.output_layers_id[-1], index_offset=-1, return_id=True
            )
            self.penultimate_layer_id = penultimate_layer_id
            penult_extractor = tf.keras.models.Model(
                new_input, penultimate_layer.output
            )
            last_layer = self.find_layer(self.model, self.output_layers_id[-1])

            # clip penultimate activations
            x = tf.clip_by_value(
                penult_extractor(new_input),
                clip_value_min=tf.float32.min,
                clip_value_max=self.react_threshold,
            )
            # apply ultimate layer on clipped activations
            output_tensors[-1] = last_layer(x)

        extractor = tf.keras.models.Model(new_input, output_tensors)
        return extractor

    @sanitize_input
    @tf.function
    def predict_tensor(self, tensor: TensorType) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Get the projection of tensor in the feature space of self.model

        Args:
            tensor (TensorType): input tensor (or dataset elem)

        Returns:
            tf.Tensor: features
        """
        features = self.extractor(tensor, training=False)
        return features

    def predict(
        self, dataset: Union[ItemType, tf.data.Dataset], **kwargs
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (Union[ItemType, tf.data.Dataset]): input dataset
            kwargs (dict): additional arguments not considered for prediction

        Returns:
            List[tf.Tensor]: features
        """
        if isinstance(dataset, get_args(ItemType)):
            tensor = TFDataHandler.get_input_from_dataset_item(dataset)
            return self.predict_tensor(tensor)

        features = [None for i in range(len(self.output_layers_id))]
        for elem in dataset:
            tensor = TFDataHandler.get_input_from_dataset_item(elem)
            features_batch = self.predict_tensor(tensor)
            if len(features) == 1:
                features_batch = [features_batch]
            for i, f in enumerate(features_batch):
                features[i] = (
                    f if features[i] is None else tf.concat([features[i], f], axis=0)
                )

        # No need to return a list when there is only one input layer
        if len(features) == 1:
            features = features[0]
        return features

    def get_weights(self, layer_id: Union[int, str]) -> List[tf.Tensor]:
        """Get the weights of a layer

        Args:
            layer_id (Union[int, str]): layer identifier

        Returns:
            List[tf.Tensor]: weights and biases matrixes
        """
        return self.find_layer(self.model, layer_id).get_weights()
