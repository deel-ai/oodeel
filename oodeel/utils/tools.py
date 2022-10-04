import tensorflow as tf

def dataset_length(
    dataset: tf.data.Dataset
) -> int:

    for x in dataset.take(1):
        if isinstance(x, tuple):
            length = len(x)
        else:
            length = 1

    return length