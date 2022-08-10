import numpy as np
import sys
import tensorflow as tf

def get_model(model_name, id_dataset="perso"):
    """
    Load a model trained on ID dataset denoted "id_dataset". 

    Parameters
    ----------
    model_name : _type_
        _description_
    id_dataset : str, optional
        if "id_dataset"==perso, load a model found on the path "model_name". 
        One other possible value: imagenet. In that last case pretrained model 
        called "model_name" is loaded from keras.applications. The str 
        "model_name" must match the name of the desired class of keras.applications.
        by default "perso"

    Returns
    -------
    keras model
    """
    assert id_dataset in ["perso", "imagenet"], "id_dataset can only be \"perso\" or \"imagenet\""
    if id_dataset != "perso":
        model = getattr(tf.keras.applications, model_name)(
                include_top=True, weights=id_dataset
            )
    else:
        model = tf.keras.models.load_model(model_name)
    return model

