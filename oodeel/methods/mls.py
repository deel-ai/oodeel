import tensorflow as tf
from .base import OODModel
import numpy as np

class MLS(OODModel):

    def __init__(self, model):
        super().__init__(model)

        feature_extractor = tf.keras.models.clone_model(model)
        feature_extractor.set_weights(model.get_weights())
        feature_extractor.layers[-1].activation = tf.keras.activations.linear
        self.feature_extractor = feature_extractor

    def score(self, inputs):
        pred = self.feature_extractor.predict(inputs)
        scores = np.max(pred, axis=1)
        self.scores = scores
        return scores

        