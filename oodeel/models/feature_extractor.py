import numpy as np
import tensorflow as tf
from ..utils.tf_operations import batch_tensor, find_layer, gradient
from ..types import *


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


        self.model = tf.keras.Model(self.input_layer, [self.output_layers]) 

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