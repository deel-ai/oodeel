import numpy as np
import tensorflow as tf
from oodeel.models.feature_extractor import KerasFeatureExtractor
from oodeel.types import *
from tests import generate_model, generate_data_tfds, almost_equal



def test_feature_extractor():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples
        )#.batch(samples)

    model = generate_model(
        input_shape=input_shape, output_shape=num_labels
        )

    model_fe = KerasFeatureExtractor(
        model, output_layers_id=[]
        )

    feature_extractor = KerasFeatureExtractor(
        model, output_layers_id=[-3]
        )

    last_layer = KerasFeatureExtractor(
        model, output_layers_id=[], input_layer_id=-2
        )

    pred_model = model.predict(data.batch(samples))
    pred_model_fe = model_fe.predict(data)
    pred_feature_extractor = feature_extractor.predict(data)
    pred_last_layer = last_layer.predict(pred_feature_extractor[0])

    assert almost_equal(pred_model, pred_model_fe)
    assert almost_equal(pred_model, pred_last_layer)
