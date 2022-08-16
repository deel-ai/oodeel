import tensorflow as tf
import numpy as np

def dataset_load(dataset_name):
    """
    Loads a dataset

    Parameters
    ----------
    dataset_name : str
    """
    assert dataset_name in ["mnist", "fashion_mnist"]
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, dataset_name).load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    return (x_train, y_train), (x_test, y_test)
