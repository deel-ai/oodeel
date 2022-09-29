import tensorflow as tf
from typing import Union, Tuple, List, Callable, Dict, Optional, Any


def batch_tensor(tensors: Union[Tuple, tf.Tensor],
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
    if not isinstance(tensors, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
    else:
        dataset = tensors
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset
