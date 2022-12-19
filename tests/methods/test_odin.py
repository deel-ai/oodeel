import tensorflow as tf
from oodeel.methods import ODIN
import numpy as np
from oodeel.types import *
from tests import generate_model, generate_data
from oodeel.datasets import DataHandler


def test_odin():
    """
    Test ODIN
    """
    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data(
        x_shape=input_shape, num_labels=num_labels, samples=samples, one_hot=False
    )  # .batch(samples)

    model = generate_model(input_shape=input_shape, output_shape=num_labels)

    odin = ODIN(temperature=100, noise=0.1)
    odin.fit(model)
    scores = odin.score(data)

    assert scores.shape == (100,)

    data = tf.data.Dataset.from_tensor_slices(data)
    scores = odin.score(data)

    assert scores.shape == (100,)
