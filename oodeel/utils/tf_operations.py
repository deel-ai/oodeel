import tensorflow as tf
from ..types import *


def batch_tensor(tensors: Union[tf.data.Dataset, tf.Tensor],
                 batch_size: Optional[int] = None):
    """
    Create a tensorflow dataset of tensors or series of tensors.

    Parameters
    ----------
    tensors
        Tuple of tensors or tensors to batch.
    batch_size
        Number of samples to iterate at once, if None process all at once.

    Returns
    -------
    dataset
        Tensorflow dataset batched.
    """
    if isinstance(tensors, tf.data.Dataset):
        for x in tensors.take(1):
            if isinstance(x, tuple):
                length = len(x)
            else:
                length = 1
        if length == 2: #when image, label
            dataset = tensors.map(lambda x, y: x)
        if length == 3: #when image, label, ood_label
            dataset = tensors.map(lambda x, y, z: x)
        
    else:
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset


def find_layer(model: tf.keras.Model, layer: Union[str, int]) -> tf.keras.layers.Layer:
    """
    Find a layer in a model either by his name or by his index.
    Parameters
    ----------
    model
        Model on which to search.
    layer
        Layer name or layer index
    Returns
    -------
    layer
        Layer found
    """
    if isinstance(layer, str):
        return model.get_layer(layer)
    if isinstance(layer, int):
        return model.layers[layer]
    raise ValueError(f"Could not find any layer {layer}.")


@tf.function
def gradient(model: Callable,
             inputs: tf.Tensor,
             targets: tf.Tensor) -> tf.Tensor:
    """
    Compute gradients for a batch of samples.
    Parameters
    ----------
    model
        Model used for computing gradient.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    Returns
    -------
    gradients
        Gradients computed, with the same shape as the inputs.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape: # type: ignore
        tape.watch(inputs)
        score = tf.reduce_sum(tf.multiply(model(inputs), targets), axis=1)
    return tape.gradient(score, inputs)