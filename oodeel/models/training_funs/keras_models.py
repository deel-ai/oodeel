import tensorflow as tf
from ...types import *
from ...utils import dataset_image_shape, dataset_cardinality
from tensorflow import keras
from keras.layers import Dense, Flatten
import numpy as np

def train_keras_app(
    train_data: tf.data.Dataset,
    model_name: str, 
    batch_size: int = 128, 
    epochs: int = 50,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    metrics: List[str] = ["accuracy"],
    imagenet_pretrained: bool = False,
    validation_data: Optional[tf.data.Dataset] = None
) -> tf.keras.Model:
    """
    _summary_

    Args:
        train_data: _description_
        model_name: _description_
        batch_size: _description_. Defaults to 128.
        epochs: _description_. Defaults to 50.
        loss: _description_. Defaults to "categorical_crossentropy".
        optimizer: _description_. Defaults to "adam".
        metrics: _description_. Defaults to ["accuracy"].
        imagenet_pretrained: _description_. Defaults to False.

    Returns:
        _description_
    """
    if imagenet_pretrained:
        model = getattr(tf.keras.applications, model_name)(
            include_top=True, weights="imagenet"
        )
        num_classes = 1000
    else:
        input_shape = dataset_image_shape(train_data)

        if input_shape == (224, 224, 3):
            model = getattr(tf.keras.applications, model_name, weights=None)()
        else:
            classes = train_data.map(lambda x, y: y).unique()
            num_classes=len(list(classes.as_numpy_iterator()))

            backbone = getattr(tf.keras.applications, model_name)(
                include_top=False, input_shape=input_shape
                )

            features = Flatten()(backbone.layers[-1].output)
            output = Dense(num_classes, activation="softmax")(features)    
            model = tf.keras.Model(backbone.layers[0].input, output)

    train_data = train_data.map(
        lambda x, y: (x, tf.one_hot(y, num_classes))
        ).batch(batch_size)

    if validation_data is not None:
        validation_data = validation_data.map(
            lambda x, y: (x, tf.one_hot(y, num_classes))
            ).batch(batch_size)

    n_steps = dataset_cardinality(train_data) * epochs 
    values = list(learning_rate * np.array([1, 0.1, 0.01]))
    boundaries = list(np.round(n_steps * np.array([1/3, 2/3])).astype(int))
    
    #### TODO 
    # Add preprocessing (data augmentation)

    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)

    config = {
        "class_name": optimizer, 
        "config": {
            "learning_rate": lr_scheduler
            }
        }

    keras_optimizer = tf.keras.optimizers.get(config)

    model.compile(loss=loss, optimizer=keras_optimizer, metrics=metrics)

    model.fit(
        train_data, 
        validation_data=validation_data, 
        epochs=epochs) 

    return model  