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
        feature_layers_id: list of str or int that identify features to output.
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
        feature_layers_id: List[Union[int, str]] = [-1],
        input_layer_id: Optional[Union[int, str]] = None,
        react_threshold: Optional[float] = None,
    ):
        if input_layer_id is None:
            input_layer_id = 0
        super().__init__(
            model=model,
            feature_layers_id=feature_layers_id,
            input_layer_id=input_layer_id,
            react_threshold=react_threshold,
        )

        self.backend = "tensorflow"
        self.model.layers[-1].activation = getattr(tf.keras.activations, "linear")
        self._last_logits = None

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
            self.find_layer(self.model, id).output for id in self.feature_layers_id
        ]

        # === If react method, clip activations from penultimate layer ===
        if self.react_threshold is not None:
            penultimate_layer = self.find_layer(self.model, -2)
            penult_extractor = tf.keras.models.Model(
                new_input, penultimate_layer.output
            )
            last_layer = self.find_layer(self.model, -1)

            # clip penultimate activations
            x = tf.clip_by_value(
                penult_extractor(new_input),
                clip_value_min=tf.float32.min,
                clip_value_max=self.react_threshold,
            )
            # apply ultimate layer on clipped activations
            output_tensors.append(last_layer(x))
        else:
            output_tensors.append(self.find_layer(self.model, -1).output)

        extractor = tf.keras.models.Model(new_input, output_tensors)
        return extractor

    @sanitize_input
    def predict_tensor(
        self,
        tensor: TensorType,
        postproc_fns: Optional[List[Callable]] = None,
    ) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """Get the projection of tensor in the feature space of self.model

        Args:
            tensor (TensorType): input tensor (or dataset elem)
            postproc_fns (Optional[List[Callable]]): postprocessing function to apply to
                each feature immediately after forward. Default to None.

        Returns:
            Tuple[List[tf.Tensor], tf.Tensor]: features, logits
        """
        features = self.forward(tensor)

        if type(features) is not list:
            features = [features]

        # split features and logits
        logits = features.pop()

        if postproc_fns is not None:
            features = [
                postproc_fn(feature)
                for feature, postproc_fn in zip(features, postproc_fns)
            ]

        self._last_logits = logits
        return features, logits

    @tf.function
    def forward(self, tensor: TensorType) -> List[tf.Tensor]:
        return self.extractor(tensor, training=False)

    def predict(
        self,
        dataset: Union[ItemType, tf.data.Dataset],
        postproc_fns: Optional[List[Callable]] = None,
        **kwargs,
    ) -> Tuple[List[tf.Tensor], dict]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (Union[ItemType, tf.data.Dataset]): input dataset
            postproc_fns (Optional[Callable]): postprocessing function to apply to each
                feature immediately after forward. Default to None.
            kwargs (dict): additional arguments not considered for prediction

        Returns:
            List[tf.Tensor], dict: features and extra information (logits, labels) as a
                dictionary.
        """
        labels = None

        if isinstance(dataset, get_args(ItemType)):
            tensor = TFDataHandler.get_input_from_dataset_item(dataset)
            features, logits = self.predict_tensor(tensor, postproc_fns)

            # Get labels if dataset is a tuple/list
            if isinstance(dataset, (list, tuple)):
                labels = TFDataHandler.get_label_from_dataset_item(dataset)

        else:  # if dataset is a tf.data.Dataset
            features = [None for i in range(len(self.feature_layers_id))]
            logits = None
            contains_labels = TFDataHandler.get_item_length(dataset) > 1
            for elem in dataset:
                tensor = TFDataHandler.get_input_from_dataset_item(elem)
                features_batch, logits_batch = self.predict_tensor(tensor, postproc_fns)

                for i, f in enumerate(features_batch):
                    features[i] = (
                        f
                        if features[i] is None
                        else tf.concat([features[i], f], axis=0)
                    )
                # concatenate logits
                logits = (
                    logits_batch
                    if logits is None
                    else tf.concat([logits, logits_batch], axis=0)
                )
                # concatenate labels of current batch with previous batches
                if contains_labels:
                    lbl_batch = TFDataHandler.get_label_from_dataset_item(elem)

                    if labels is None:
                        labels = lbl_batch
                    else:
                        labels = tf.concat([labels, lbl_batch], axis=0)

        # store extra information in a dict
        info = dict(labels=labels, logits=logits)
        return features, info

    def get_weights(self, layer_id: Union[int, str]) -> List[tf.Tensor]:
        """Get the weights of a layer

        Args:
            layer_id (Union[int, str]): layer identifier

        Returns:
            List[tf.Tensor]: weights and biases matrixes
        """
        return self.find_layer(self.model, layer_id).get_weights()
