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
    def __init__(self, model, indices):
        if type(indices) is not list:
            indices = [indices]
        self.indices = indices
        self.extractors = []
        self.flatten=None
        
        markers = [0] + indices
        for i in range(1, len(markers)):
            self.extractors.append(
                Sequential(model.layers[markers[i-1]:markers[i]]))



    def predict(self, inputs, flatten=True):
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
        np.array
            projected samples
        """
        predictions = []
        for ext in self.extractors:
            inputs = ext.predict(inputs)
            if flatten:
                self.flatten = flatten
                predictions.append(inputs.reshape(-1, inputs.shape[-1]))
            else:
                predictions.append(inputs)
        return predictions

    def __call__(self, inputs):
        """
        Convenience wrapper for predict
        """
        return self.predict(inputs)