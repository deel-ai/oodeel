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
import tensorflow_probability as tfp
from tqdm import tqdm

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
    (logits).

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
            when working on the feature space without finetuning the bottom of the
            model).
            Defaults to None.
        react_threshold: if not None, penultimate layer activations are clipped under
            this threshold value (useful for ReAct). Defaults to None.
        scale_percentile: if not None, the features are scaled
            following the method of Xu et al., ICLR 2024.
            Defaults to None.
        ash_percentile: if not None, the features are scaled following
            the method of Djurisic et al., ICLR 2023.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        feature_layers_id: List[Union[int, str]] = [],
        head_layer_id: Optional[Union[int, str]] = -1,
        input_layer_id: Optional[Union[int, str]] = None,
        react_threshold: Optional[float] = None,
        scale_percentile: Optional[float] = None,
        ash_percentile: Optional[float] = None,
    ):
        if input_layer_id is None:
            input_layer_id = 0
        super().__init__(
            model=model,
            feature_layers_id=feature_layers_id,
            input_layer_id=input_layer_id,
            head_layer_id=head_layer_id,
            react_threshold=react_threshold,
            scale_percentile=scale_percentile,
            ash_percentile=ash_percentile,
        )

        self.backend = "tensorflow"
        self.model.layers[-1].activation = getattr(tf.keras.activations, "linear")
        self._last_logits = None

    @staticmethod
    def find_layer(
        model: tf.keras.Model,
        layer_id: Union[str, int],
        index_offset: int = 0,
        return_id: bool = False,
    ) -> Union[tf.keras.layers.Layer, Tuple[tf.keras.layers.Layer, str]]:
        """Find a layer in a model either by his name or by his index.

        Args:
            model (tf.keras.Model): model whose identified layer will be returned
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

    @staticmethod
    def get_layer_index_by_name(model: tf.keras.Model, layer_id: str) -> int:
        """Get the index of a layer by its name.

        Args:
            model (tf.keras.Model): The model containing the layers.
            layer_id (str): The name of the layer.

        Returns:
            int: The index of the layer.

        Raises:
            ValueError: If the layer with the given name is not found.
        """
        layers_names = [layer.name for layer in model.layers]
        if layer_id not in layers_names:
            raise ValueError(f"Layer with name '{layer_id}' not found in the model.")
        return layers_names.index(layer_id)

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
            penultimate_layer = self.find_layer(
                self.model, self.head_layer_id, index_offset=-1
            )
            penult_extractor = tf.keras.models.Model(
                new_input, penultimate_layer.output
            )
            last_layer = self.find_layer(self.model, self.head_layer_id)

            # clip penultimate activations
            x = tf.clip_by_value(
                penult_extractor(new_input),
                clip_value_min=tf.float32.min,
                clip_value_max=self.react_threshold,
            )
            # apply ultimate layer on clipped activations
            output_tensors.append(last_layer(x))

        # === If SCALE method, scale activations from penultimate layer ===
        # === If ASH method, scale and prune activations from penultimate layer ===
        elif (self.scale_percentile is not None) or (self.ash_percentile is not None):
            penultimate_layer = self.find_layer(
                self.model, self.head_layer_id, index_offset=-1
            )
            penult_extractor = tf.keras.models.Model(
                new_input, penultimate_layer.output
            )
            last_layer = self.find_layer(self.model, self.head_layer_id)

            # apply scaling on penultimate activations
            penultimate = penult_extractor(new_input)
            if self.scale_percentile is not None:
                output_percentile = tfp.stats.percentile(
                    penultimate, 100 * self.scale_percentile, axis=1
                )
            else:
                output_percentile = tfp.stats.percentile(
                    penultimate, 100 * self.ash_percentile, axis=1
                )

            mask = penultimate > tf.reshape(output_percentile, (-1, 1))
            filtered_penultimate = tf.where(
                mask, penultimate, tf.zeros_like(penultimate)
            )
            s = tf.math.exp(
                tf.reduce_sum(penultimate, axis=1)
                / tf.reduce_sum(filtered_penultimate, axis=1)
            )

            if self.scale_percentile is not None:
                x = penultimate * tf.expand_dims(s, 1)
            else:
                x = filtered_penultimate * tf.expand_dims(s, 1)
            # apply ultimate layer on scaled activations
            output_tensors.append(last_layer(x))

        else:
            output_tensors.append(
                self.find_layer(self.model, self.head_layer_id).output
            )

        extractor = tf.keras.models.Model(new_input, output_tensors)
        return extractor

    @sanitize_input
    def predict_tensor(
        self,
        tensor: TensorType,
        postproc_fns: Optional[List[tf.keras.Model]] = None,
    ) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """Get the projection of tensor in the feature space of self.model

        Args:
            tensor (TensorType): input tensor (or dataset elem)
            postproc_fns (Optional[List[tf.keras.Model]]): postprocessing function to apply to
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
        postproc_fns: Optional[List[tf.keras.Model]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[List[tf.Tensor], dict]:
        """Get the projection of the dataset in the feature space of self.model

        Args:
            dataset (Union[ItemType, tf.data.Dataset]): input dataset
            postproc_fns (Optional[tf.keras.Model]): postprocessing function to apply to each
                feature immediately after forward. Default to None.
            verbose (bool): if True, display a progress bar. Defaults to False.
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
            for elem in tqdm(dataset, desc="Predicting", disable=not verbose):
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
