import tensorflow as tf


def load_dataset(dataset_name):
    """
    Loads a dataset

    Parameters
    ----------
    dataset_name : str
    """
    assert dataset_name in ["mnist", "fashion_mnist"]
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, dataset_name).load_data()
    return (x_train, y_train), (x_test, y_test)
