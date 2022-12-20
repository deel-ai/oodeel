import tensorflow as tf
from oodeel.methods import DKNN
import numpy as np
from oodeel.types import *
from tests import generate_model, generate_data, almost_equal


def test_dknn():
    """
    Test DKNN
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data_x, _ = generate_data(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )  # .batch(samples)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    dknn = DKNN()
    dknn.fit(model, fit_dataset=data_x[:100])
    scores = dknn.score(data_x)

    assert scores.shape == (100,)

    data = tf.data.Dataset.from_tensor_slices(data_x)
    scores = dknn.score(data)

    assert scores.shape == (100,)
