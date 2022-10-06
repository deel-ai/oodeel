import tensorflow as tf
import numpy as np
from oodeel.datasets import DataHandler
from tests import almost_equal, generate_data_tfds
from oodeel.utils import dataset_nb_columns, dataset_max_pixel


"""
def test_load_tfds():
    data_handler = DataHandler()
    ds = data_handler.load_tfds('mnist', preprocess=True)
    (x_train, y_train),  (x_test, y_test) = data_handler.load_tfds(
        'mnist', 
        preprocess=True,
        as_numpy=True
    )

    x_tf = ds["train"]
    max_val_tf = dataset_max_pixel(x_tf)
    max_val_np = np.max(x_train)
    assert isinstance(ds["train"], tf.data.Dataset)
    assert max_val_np == 1.0
    assert max_val_tf == 1.0
"""

def test_convert_to_numpy():
    data_handler = DataHandler()
    ds = generate_data_tfds(
        x_shape=(32, 32, 3), num_labels=10, samples=100,
        one_hot=False
        )
    x, y = data_handler.convert_to_numpy(ds)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == (100, 32, 32, 3)
    assert y.shape == (100,)

def test_filter_tfds():
    data_handler = DataHandler()
    inc_labels = [0, 1, 2, 3, 4]
    ds = generate_data_tfds(
        x_shape=(32, 32, 3), num_labels=10, samples=100,
        one_hot=False
        )
    data_id, data_ood = data_handler.filter_tfds(ds, inc_labels = inc_labels)
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
    data_id = generate_data_tfds(x_shape=(32, 32, 3), num_labels=10, samples=100)
    data_ood = generate_data_tfds(x_shape=(32, 32, 3), num_labels=10, samples=100)

    data = data_handler.merge_tfds(data_id, data_ood, shuffle='True')
    labels = data_handler.get_ood_labels(data)

    x, y, z = data_handler.convert_to_numpy(data)

    assert np.sum(labels[:100]) not in [0, 100]
    assert x.shape[0] == 200
    assert y.shape[0] == 200
    assert np.sum(z - labels) == 0.0




