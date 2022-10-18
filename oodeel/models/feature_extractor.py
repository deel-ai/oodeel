import numpy as np
import tensorflow as tf
from ..utils.tf_operations import batch_tensor, find_layer, gradient
from ..types import *


class KerasFeatureExtractor(object):
    """
    Feature extractor based on "model" to construct a feature space 
    on which OOD detection is performed. The features are the output
    activation values of internal model layers. 

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
        output_layers_id: List[Union[int, str]] =[], 
        output_activation: str = None, 
        input_layer_id : Union[int, str] = None,
        flatten: bool = False, 
        batch_size: int = 256,
    ):
        if type(output_layers_id) is not list:
            output_layers_id = [output_layers_id]

        self.output_layers_id = output_layers_id
        self.output_activation = output_activation
        self.flatten=flatten
        self.batch_size=batch_size

        self.output_layers = [find_layer(model, ol_id).output for ol_id in output_layers_id] 
        if self.output_activation is not None:
            model.layers[-1].activation = getattr(tf.keras.activations, self.output_activation)      

        self.output_layers.append(model.output)

        if input_layer_id is not None:
            self.input_layer = find_layer(model, input_layer_id).input
        else:
            self.input_layer = model.input

        self.model = self.construct_model()


    #@tf.function
    def construct_model(self):
        model = tf.keras.Model(self.input_layer, [self.output_layers]) 
        return model


    def predict(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """
        Projects input samples "inputs" into the feature space

        Args:
            inputs: input samples to project in feature space

        Returns:
            list of features
        """
        assert not isinstance(inputs, tuple), "Inputs must be tf.data.Dataset, tensor or arrays (not tuples)"

        inputs = batch_tensor(inputs, self.batch_size)
        features = self.model.predict(inputs)

        if len(self.output_layers_id) > 0: 
            features = features[0]

        if self.flatten:
            features = [feature.reshape(feature.shape[0], -1) for feature in features]
        return features

    def __call__(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """
        Convenience wrapper for predict
        """
        return self.predict(inputs)

