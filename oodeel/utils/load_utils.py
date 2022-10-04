import numpy as np
import sys
import tensorflow as tf
from ..types import *

def get_model(
    model_name: str, 
    id_dataset: Optional[str] = None
):
    """
    Loads a model trained on ID dataset denoted "id_dataset". 

    Parameters
    ----------
    model_name : _type_
        _description_
    id_dataset : str, optional
        if "id_dataset" is None, load a model found on the path "model_name". 
        One other possible value: imagenet. In that last case pretrained model 
        called "model_name" is loaded from keras.applications. The str 
        "model_name" must match the name of the desired class of keras.applications.
        by default None

    Returns
    -------
    keras model
    """
    assert id_dataset in [None, "imagenet"], "id_dataset can only be None or \"imagenet\""
    if id_dataset is not None:
        model = getattr(tf.keras.applications, model_name)(
                include_top=True, weights=id_dataset
            )
    else:
        model = tf.keras.models.load_model(model_name)
    return model


def keras_dataset_load(
    dataset_name: str
) -> Tuple[Tuple[Union[tf.data.Dataset, tf.Tensor, np.ndarray]]]:
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
