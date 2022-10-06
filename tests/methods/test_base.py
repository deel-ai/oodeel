from typing import Type, Union, Iterable, Callable
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from oodeel.methods import MLS
from oodeel.types import *
from tests import generate_data_tfds, generate_model, almost_equal
"""
def test_isood():

    input_shape = (32, 32, 3)
    num_labels = 10
    samples = 100

    data = generate_data_tfds(
        x_shape=input_shape, num_labels=num_labels, samples=samples,
        one_hot=False
        )

    model = generate_model(
        input_shape=input_shape, output_shape=num_labels
        )

    oodmodel = MLS()
    oodmodel.fit(model)

    oodmodel.threshold = -5
    isooddata = oodmodel.isood(inputs=data)
    isooddata2 = oodmodel(data)
    print(np.sum(isooddata - isooddata2))

    assert almost_equal(isooddata, isooddata2)
"""