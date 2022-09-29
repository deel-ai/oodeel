import numpy as np
import tensorflow as tf
from ..utils.tf_operations import batch_tensor, find_layer, gradient
from ..types import *


#class ThomasFeatureExtractor(object):

class KerasFeatureExtractor(object):
    """
    Feature extractor based on "model" to construct a feature space 
    on which OOD detection is performed. The features are the output
    activation values of internal model layers. 

    Parameters
    ----------
    model : keras model
        model used to build the feature space
    indices : list of int
        indices of the internal layers to get the activation values from
    """
    def __init__(
        self, 
        model: Callable, 
        output_layers: List[int] =[-1], 
        output_activations: List[str] = ["base"], 
        flatten: bool = True, 
        batch_size: int = 256,
    ):

        self.model = model

        if type(output_layers) is not list:
            output_layers = [output_layers]
        self.output_layers = output_layers
        
        if type(output_activations) is not list:
            output_activations = [output_activations]
        
        self.output_activations = output_activations
        self.flatten=flatten
        self.batch_size=batch_size
        

    def predict(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """
        Projects input samples "inputs" into the constructed feature space

        Parameters
        ----------
        inputs : np.array
            input samples to project 
        flatten : bool, optional
            whether to flatten the activation outputs or not, by default True

        Returns
        -------
        list
            list of outputs of selected output layers
        """
        features = None
        assert not isinstance(inputs, tuple), "Inputs must be tf.data.Dataset, tensor or arrays (not tuples)"

        for inputs_batched in batch_tensor(inputs, self.batch_size):
            inputs_b = tf.cast(inputs_batched, tf.float32)
            features_batch = None
            ind_output=0
            for i, layer in enumerate(self.model.layers):
                # Store the input if needed (must be done here to avoid duplicating layer)
                if i -len(self.model.layers) in self.output_layers:
                    input_out = tf.Variable(inputs_b)

                inputs_b = layer(inputs_b)

                if i - len(self.model.layers) in self.output_layers:

                    if self.output_activations[ind_output] != "base":        
                        activation = getattr(tf.keras.activations, self.output_activations[ind_output])
                        layer.activation = activation
                    output = layer(input_out)

                    if self.flatten:
                        dim = tf.reduce_prod(output.shape[1:])
                        output = tf.reshape(output, [-1, dim])

                    features_batch = output if features_batch is None else tf.concat(
                        [features_batch, output], axis=1
                    )
                    ind_output += 1

            features = features_batch if features is None else tf.concat(
                        [features, features_batch], axis=0
            )
        return features

    def __call__(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        """
        Convenience wrapper for predict
        """
        return self.predict(inputs)


class TorchFeatureExtractor(object):
    """
    Feature extractor based on "model" to construct a feature space 
    on which OOD detection is performed. The features are the output
    activation values of internal model layers. 

    Parameters
    ----------
    model : keras model
        model used to build the feature space
    indices : list of int
        indices of the internal layers to get the activation values from
    """
    def __init__(
        self, 
        model, 
        output_layers=[-1], 
        output_activations=["base"], 
        flatten=True, 
        batch_size=256
    ):

        self.model = model
        if type(output_layers) is not list:
            output_layers = [output_layers]
        self.output_layers = output_layers
        if type(output_activations) is not list:
            output_activations = [output_activations]
        self.output_activations = output_activations
        self.flatten=flatten
        self.batch_size=batch_size
        

    def predict(self, inputs):
        """
        Projects input samples "inputs" into the constructed feature space

        Parameters
        ----------
        inputs : np.array
            input samples to project 
        flatten : bool, optional
            whether to flatten the activation outputs or not, by default True

        Returns
        -------
        list
            list of outputs of selected output layers
        """
        raise NotImplementedError()

    def __call__(self, inputs):
        """
        Convenience wrapper for predict
        """
        return self.predict(inputs)