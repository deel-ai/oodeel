import tensorflow as tf
import numpy as np

        

def keras_dataset_load(dataset_name):
    """
    Loads a dataset

    Parameters
    ----------
    dataset_name : str
    """
    assert hasattr(tf.keras.datasets, dataset_name), f"{dataset_name} not available with keras.datasets"
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, dataset_name).load_data()

    x_max = np.max(x_train)
    x_train = x_train.astype("float32") / x_max
    x_test = x_test.astype("float32") / x_max

    if dataset_name in ["mnist", "fashion_mnist"]:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    return (x_train, y_train), (x_test, y_test)
