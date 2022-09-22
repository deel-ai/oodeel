import numpy as np
import sys
import tensorflow as tf

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

