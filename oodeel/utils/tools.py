import tensorflow as tf
from oodeel.types import *

def dataset_nb_columns(
    dataset: tf.data.Dataset
) -> int:

    for x in dataset.take(1):
        if isinstance(x, tuple):
            length = len(x)
        else:
            length = 1

    return length

def dataset_image_shape(
    dataset: tf.data.Dataset
) -> Tuple[int]:

    for x in dataset.take(1):
        if isinstance(x, tuple):
            shape = x[0].shape
        else:
            shape = x.shape

    return shape

def dataset_max_pixel(
    dataset: tf.data.Dataset
) -> Tuple[int]:

    length = dataset_nb_columns(dataset)

    if length == 2:
        dataset = dataset.map(lambda x, y: x)
    elif length == 3:
        dataset = dataset.map(lambda x, y, z: x)

    max_pixel = dataset.reduce(
        0., lambda x, y: float(tf.math.reduce_max(tf.maximum(x, y)))
        )
    return float(max_pixel)

