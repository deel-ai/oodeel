import tensorflow as tf
import numpy as np
from oodeel.datasets import DataHandler
from tests import almost_equal
from oodeel.utils import dataset_length




def test_load_tfds():
    data_handler = DataHandler()
    ds = data_handler.load_tfds('mnist', preprocess=True)
    (x_train, y_train),  (x_test, y_test) = data_handler.load_tfds(
        'mnist', 
        preprocess=True,
        as_numpy=True
    )

    x_tf = ds["train"].map(lambda x, y: x)
    max_val_tf = x_tf.reduce(0., lambda x, y: float(tf.math.reduce_max(tf.maximum(x, y))))
    max_val_np = np.max(x_train)
    assert isinstance(ds["train"], tf.data.Dataset)
    assert max_val_np == 1.0
    assert max_val_tf == 1.0


def test_convert_to_numpy():
    data_handler = DataHandler()
    ds = data_handler.load_tfds('mnist')
    x, y = data_handler.convert_to_numpy(ds["test"])
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == (10000, 28, 28, 1)
    assert y.shape == (10000,)

def test_filter_tfds():
    data_handler = DataHandler()
    inc_labels = [0, 1, 2, 3, 4]
    ds = data_handler.load_tfds('mnist')
    data_id, data_ood = data_handler.filter_tfds(ds["test"], inc_labels = inc_labels)
    x_id, y_id = data_handler.convert_to_numpy(data_id)
    x_ood, y_ood = data_handler.convert_to_numpy(data_ood)

    lab_id = np.unique(y_id)
    lab_ood = np.unique(y_ood)
    assert len(lab_id) == 5
    assert len(lab_ood) == 5
    assert np.all([y in inc_labels for y in lab_id])
    assert np.all([y not in inc_labels for y in lab_ood])

def test_merge_tfds_get_ood_labels():
    data_handler = DataHandler()
    ds1 = data_handler.load_tfds('mnist')
    ds2 = data_handler.load_tfds('fashion_mnist')
    data_id = ds1["test"]
    data_ood = ds2["test"]

    data = data_handler.merge_tfds(data_id.take(1000), data_ood.take(1000), shuffle='True')
    labels = data_handler.get_ood_labels(data)

    x, y, z = data_handler.convert_to_numpy(data)

    assert np.sum(labels[:1000]) not in [0, 1000]
    assert x.shape[0] == 2000
    assert y.shape[0] == 2000
    assert np.sum(z - labels) == 0.0




