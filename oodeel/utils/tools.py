import tensorflow as tf
from oodeel.types import *


def dataset_nb_columns(dataset: tf.data.Dataset) -> int:

    for x in dataset.take(1):
        if isinstance(x, tuple):
            length = len(x)
        else:
            length = 1

    return length


def dataset_image_shape(dataset: tf.data.Dataset) -> Tuple[int]:
    """
    Get the shape of the images in the dataset

    Args:
        dataset:   input dataset
    Returns:
        shape of the images in the dataset
    """
    for x in dataset.take(1):
        if isinstance(x, tuple):
            shape = x[0].shape
        else:
            shape = x.shape
    return shape

def dataset_label_shape(dataset: tf.data.Dataset) -> Tuple[int]:

    for x in dataset.take(1):
        assert len(x) > 1, "No label to get the shape from"
        shape = x[1].shape
    return shape


def dataset_max_pixel(dataset: tf.data.Dataset) -> float:

    dataset = dataset_get_columns(dataset, 0)
    max_pixel = dataset.reduce(
        0.0, lambda x, y: float(tf.math.reduce_max(tf.maximum(x, y)))
    )
    return float(max_pixel)


def dataset_nb_labels(dataset: tf.data.Dataset) -> int:

    dataset = dataset_get_columns(dataset, 1)
    max_label = dataset.reduce(0, lambda x, y: int(tf.maximum(x, y)))
    return int(max_label) + 1


def dataset_cardinality(dataset: tf.data.Dataset) -> int:

    cardinality = dataset.reduce(0, lambda x, _: x + 1)
    return int(cardinality)


def dataset_get_columns(
    dataset: tf.data.Dataset, columns: Union[int, List[int]]
) -> tf.data.Dataset:
    """
    Construct a dataset out of the columns of the input dataset. The columns are identified by "columns". Here columns means x, y, or ood_labels

    Args:
        dataset: input dataset
        columns: columns to extract

    Returns:
        tf.data.Dataset with columns extracted from the input dataset
    """
    if isinstance(columns, int):
        columns = [columns]
    length = dataset_nb_columns(dataset)

    if length == 2:  # when image, label

        def return_columns(x, y, col):
            X = [x, y]
            return tuple([X[i] for i in col])

        dataset = dataset.map(lambda x, y: return_columns(x, y, columns))

    if length == 3:  # when image, label, ood_label

        def return_columns(x, y, z, col):
            X = [x, y, z]
            return tuple([X[i] for i in col])

        dataset = dataset.map(lambda x, y, z: return_columns(x, y, z, columns))

    return dataset
