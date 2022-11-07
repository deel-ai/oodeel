import numpy as np
import tensorflow as tf
from ..utils.tf_operations import batch_tensor, find_layer, gradient, gradient_single
from ..utils.tools import dataset_nb_columns
from ..types import *


class KerasFeatureExtractor(object):
    """
    Feature extractor based on "model" to construct a feature space
    on which OOD detection is performed. The features can be the output
    activation values of internal model layers, or the output of the model (softmax/logits).

    Args:
        model: model to extract the features from
        output_layers_id: list of str or int that identify features to output.
            If int, the rank of the layer in the layer list
            If str, the name of the layer.
            Defaults to [].
        output_activation: activation function for the last layer.
            Defaults to None.
        input_layer_id: input layer of the feature extractor (to avoid useless forwards
            when working on the feature space without finetuning the bottom of the model).
            Defaults to None.
        flatten: Flatten the output features or not.
            Defaults to True.
        batch_size: batch_size used to compute the features space
            projection of input data.
            Defaults to 256.
    """

    def __init__(
        self,
        model: Callable,
        output_layers_id: List[Union[int, str]] = [],
        output_activation: str = None,
        input_layer_id: Union[int, str] = None,
        flatten: bool = False,
        batch_size: int = 256,
    ):
        if type(output_layers_id) is not list:
            output_layers_id = [output_layers_id]

        self.output_layers_id = output_layers_id
        self.output_activation = output_activation
        self.flatten = flatten
        self.batch_size = batch_size

        self.output_layers = [
            find_layer(model, ol_id).output for ol_id in output_layers_id
        ]
        if self.output_activation is not None:
            if isinstance(self.output_activation, str):
                model.layers[-1].activation = getattr(
                    tf.keras.activations, self.output_activation
                )
            else:
                model.layers[-1].activation = self.output_activation

        self.output_layers.append(model.output)

        if input_layer_id is not None:
            self.input_layer = find_layer(model, input_layer_id).input
        else:
            self.input_layer = model.input

        self.model = self.construct_model()

    # @tf.function
    def construct_model(self):
        """
        Constructs the feature extractor model

        Returns:
        """
        model = tf.keras.Model(self.input_layer, [self.output_layers])
        return model

    def predict(
        self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """
        Projects input samples "inputs" into the feature space

        Args:
            inputs: input samples to project in feature space

        Returns:
            list of features
        """
        assert not isinstance(
            inputs, tuple
        ), "Inputs must be tf.data.Dataset, tensor or arrays (not tuples)"

        """
        inputs = batch_tensor(inputs, self.batch_size)
        features = self.model.predict(inputs)
        """

        features = [None for i in range(1 + len(self.output_layers_id))]

        if isinstance(inputs, tf.data.Dataset):
            nb_columns = dataset_nb_columns(inputs)

        batch_dataset = batch_tensor(inputs, self.batch_size)
        nb_columns = dataset_nb_columns(batch_dataset)
        for inputs_b in batch_dataset:
            if nb_columns > 1:
                inputs_b = inputs_b[:1][0]
            features_batch = self.model(inputs_b)
            for i, f in enumerate(features_batch):
                features[i] = (
                    f if features[i] is None else tf.concat([features[i], f], axis=0)
                )

        if len(self.output_layers_id) > 0:
            features = features[0]

        features = [
            tf.make_ndarray(tf.make_tensor_proto(feature)) for feature in features
        ]

        if self.flatten:
            features = [feature.reshape(feature.shape[0], -1) for feature in features]
        return features

    def gradient_full(self, inputs: tf.Tensor) -> tf.Tensor:
        return gradient(self.model, inputs)

    def gradient_max(
        self, inputs: Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """
        Returns the gradient wrt the dimension corresponding to the prediction. Applies after a softmax or on logits.
        TODO make this method more general: gradients wrt an arbitrary neuron

        Args:
            inputs: _description_

        Returns:
            _description_
        """
        assert isinstance(inputs, tuple) or isinstance(
            inputs, tf.data.Dataset
        ), "Inputs must be tf.data.Dataset, or a tuple of np.ndarray/tf.Tensor"

        gradients = None
        batch_dataset = batch_tensor(inputs, self.batch_size, one_hot_encode=True)

        for inputs_b in batch_dataset:
            inputs_b, outputs_b = inputs_b[:2]
            gradients_batch = gradient_single(self.model, inputs_b, outputs_b)
            gradients = (
                gradients_batch
                if gradients is None
                else tf.concat([gradients, gradients_batch], axis=0)
            )
        #
        gradients = tf.make_ndarray(tf.make_tensor_proto(gradients))

        if self.flatten:
            gradients = gradients.reshape(gradients.shape[0], -1)

        return gradients

    def __call__(
        self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """
        Convenience wrapper for predict
        """
        return self.predict(inputs)
