import tensorflow as tf
from ...types import *

def train_keras_app(
    train_data: tf.data.Dataset,
    model_name: str, 
    batch_size: int = 128, 
    epochs: int = 50,
    loss: str = "categorical_crossentropy",
    optimizer: str = "adam",
    metrics: List[str] = ["accuracy"],
    imagenet_pretrained: bool = False
) -> tf.keras.Model:

    
    weights = "imagenet" if imagenet_pretrained else None

    model = getattr(tf.keras.applications, model_name)(
        include_top=True, weights=weights
    )

    classes = train_data.map(lambda x, y: y).unique()
    num_classes=len(list(classes.as_numpy_iterator()))

    train_data = train_data.map(lambda x, y: (x, tf.one_hot(y, num_classes))).batch(batch_size)

    #### TODO 
    # Add preprocessing (data augmentation)
    # Add learning rate schedule
    # Add early stopping ?

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_data, epochs=epochs) 

    return model  