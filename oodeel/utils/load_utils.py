import numpy as np
import sys
import tensorflow as tf

def get_model(model_name, id_dataset="perso"):
    if id_dataset != "perso":
        model = getattr(tf.keras.applications, model_name)(
                include_top=True, weights=id_dataset
            )
    else:
        model = tf.keras.models.load_model(model_name)
    return model

