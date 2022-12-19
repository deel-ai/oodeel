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
    # TODO check with Thomas about @tf.function
    def construct_model(self):
        """
        Constructs the feature extractor model

        Returns:
        """
        model = tf.keras.Model(self.input_layer, self.output_layers)
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
            if len(self.output_layers) > 1:
                for i, f in enumerate(features_batch):
                    features[i] = (
                        f
                        if features[i] is None
                        else tf.concat([features[i], f], axis=0)
                    )
            else:
                features[0] = (
                    features_batch
                    if features[0] is None
                    else tf.concat([features[0], features_batch], axis=0)
                )

        features = [
            tf.make_ndarray(tf.make_tensor_proto(feature)) for feature in features
        ]

        if self.flatten:
            features = [feature.reshape(feature.shape[0], -1) for feature in features]
        return features

    def gradient_index(
        self,
        inputs: Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]],
        index: Union[list, int],
    ) -> tf.Tensor:
        """
        Computes the gradients of a specific dimension of the output, identified with index, wrt the input.

        Parameters
        ----------
        inputs : Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]]
            input tensor
        index : Union[list, int]
            indices to identify the output dimension

        Returns
        -------
        tf.Tensor
            computed gradients
        """
        assert (
            len(self.output_layers) == 1
        ), "Only one output layer is supported for gradient calculation. If gradients wrt several outputs are needed, construct one FeatureExtractor for each"

        gradients = None
        batch_dataset = batch_tensor(inputs, self.batch_size)
        nb_columns = dataset_nb_columns(batch_dataset)

        for inputs_b in batch_dataset:
            if nb_columns > 1:
                inputs_b = inputs_b[0]
            gradients_batch = gradient(self.model, inputs_b, index)
            gradients = (
                gradients_batch
                if gradients is None
                else tf.concat([gradients, gradients_batch], axis=0)
            )

        gradients = tf.make_ndarray(tf.make_tensor_proto(gradients))

        if self.flatten:
            gradients = gradients.reshape(gradients.shape[0], -1)

        return gradients

    def gradient_pred(
        self, inputs: Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """
        Returns the gradients wrt the dimension corresponding to the prediction (max output). Applies after a softmax or logits.

        Parameters
        ----------
        inputs : Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]]
            input tensor

        Returns
        -------
        tf.Tensor
            computed gradients
        """
        gradients = None
        batch_dataset = batch_tensor(inputs, self.batch_size)
        num_classes = list(self.model.layers[-1].output.shape)[1]

        for inputs_b in batch_dataset:
            if isinstance(inputs_b, tuple):
                inputs_b = inputs_b[:1]
            preds = self.model(inputs_b)
            outputs_b = tf.one_hot(tf.argmax(preds, axis=1), num_classes)
            gradients_batch = gradient_single(self.model, inputs_b, outputs_b)[0]
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

    def gradient_true_pred(
        self, inputs: Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """
        Returns the gradients wrt the dimension corresponding to the true label. Applies after a softmax or on logits.

        Parameters
        ----------
        inputs : Union[tf.data.Dataset, Tuple[tf.Tensor], Tuple[np.ndarray]]
            input dataset. Must include the input tensors and the labels.

        Returns
        -------
        tf.Tensor
            computed gradients
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
