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
import tensorflow as tf

from ..datasets import TFDataHandler
from ..types import Callable
from ..types import List
from ..types import Tuple
from ..types import Union
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
    """

    def __init__(
        self,
        model: Callable,
        output_layers_id: List[Union[int, str]] = [-1],
        input_layer_id: Union[int, str] = 0,
    ):
        super().__init__(
            model=model,
            output_layers_id=output_layers_id,
            input_layer_id=input_layer_id,
        )

        self.model.layers[-1].activation = getattr(tf.keras.activations, "linear")

    def find_layer(self, layer_id: Union[str, int]) -> tf.keras.layers.Layer:
        """Find a layer in a model either by his name or by his index.

        Args:
            layer_id (Union[str, int]): layer identifier

        Raises:
            ValueError: if the layer is not found

        Returns:
            tf.keras.layers.Layer: the corresponding layer
        """
        if isinstance(layer_id, str):
            return self.model.get_layer(layer_id)
        if isinstance(layer_id, int):
            return self.model.get_layer(index=layer_id)
        raise ValueError(f"Could not find any layer {layer_id}.")

    # @tf.function
    # TODO check with Thomas about @tf.function
    def prepare_extractor(self) -> tf.keras.models.Model:
        """Constructs the feature extractor model

        Returns:
            tf.keras.models.Model: truncated model (extractor)
        """
        output_layers = [
            self.find_layer(ol_id).output for ol_id in self.output_layers_id
        ]

        input_layer = self.find_layer(self.input_layer_id)
        new_input = tf.keras.layers.Input(tensor=input_layer.input)
        extractor = tf.keras.models.Model(new_input, output_layers)
        return extractor

    @tf.function
    def predict_tensor(self, tensor: Union[tf.Tensor, np.ndarray, Tuple]) -> tf.Tensor:
        """Get the projection of tensor in the feature space of self.model

        Args:
            tensor (Union[tf.Tensor, np.ndarray, Tuple]): input tensor (or dataset elem)

        Returns:
            tf.Tensor: features
        """
        features = self.extractor(tensor)
        return features

    def predict(self, dataset: tf.data.Dataset) -> List[tf.Tensor]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (tf.data.Dataset): input dataset

        Returns:
            List[tf.Tensor]: features
        """
        if not isinstance(dataset, tf.data.Dataset):
            tensor = TFDataHandler.get_input_from_dataset_elem(dataset)
            return self.predict_tensor(tensor)

        features = [None for i in range(len(self.output_layers_id))]
        for elem in dataset:
            tensor = TFDataHandler.get_input_from_dataset_elem(elem)
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

    def get_weights(self, layer_id: Union[int, str]) -> tf.Tensor:
        """Get the weights of a layer

        Args:
            layer_id (Union[int, str]): layer identifier

        Returns:
            tf.Tensor: weights matrix
        """
        return self.find_layer(layer_id).get_weights()
