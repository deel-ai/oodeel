import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

class FeatureExtractor(object):
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
    def __init__(self, model, output_layers=[-1], output_activations=["base"], flatten=True):
        self.model = model
        if type(output_layers) is not list:
            output_layers = [output_layers]
        self.output_layers = output_layers
        if type(output_activations) is not list:
            output_activations = [output_activations]
        self.output_activations = output_activations
        self.flatten=flatten
        

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
        features_list = []
        ind_output=0
        for i, layer in enumerate(self.model.layers):
            # Store the input if needed (must be done here to avoid duplicating layer)
            if i -len(self.model.layers) in self.output_layers:
                input_out = tf.Variable(inputs)

            inputs = layer(inputs)

            if i -len(self.model.layers) in self.output_layers:
                if self.output_activations[ind_output] != "base":        
                    activation = getattr(tf.keras.activations, self.output_activations[ind_output])
                    layer.activation = activation
                output = layer(input_out)
                if self.flatten:
                    dim = tf.reduce_prod(output.shape[1:])
                    features_list.append(np.reshape(output, [-1, dim]))
                else:
                    features_list.append(np.array(output))
                ind_output += 1

        return features_list

    def __call__(self, inputs):
        """
        Convenience wrapper for predict
        """
        return self.predict(inputs)