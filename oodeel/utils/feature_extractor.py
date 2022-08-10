import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

class FeatureExtractor(object):

    def __init__(self, model, indices, flatten=True):
        if type(indices) is not list:
            indices = [indices]
        self.indices = indices
        self.extractors = []
        self.flatten=flatten
        
        markers = [0] + indices
        print(model.layers)
        for i in range(1, len(markers)):
            self.extractors.append(
                Sequential(model.layers[markers[i-1]:markers[i]]))



    def predict(self, inputs):
        predictions = []
        for ext in self.extractors:
            inputs = ext.predict(inputs)
            if self.flatten:
                predictions.append(inputs.flatten())
            else:
                predictions.append(inputs)
        return predictions

    def __call__(self, inputs):
        return self.predict(inputs)