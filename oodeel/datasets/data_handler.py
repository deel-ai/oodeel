import tensorflow as tf
import numpy as np
from ..utils.load_utils import keras_dataset_load
from ..types import *
from ..utils import dataset_nb_columns
import tensorflow_datasets as tfds
import os

class DataHandler(object):
    """
    Handles datasets (filtering by labels for now).
    Aims at handling datasets from diverse sources

    Args:
        x: inputs
        y: labels
    """
    def __init__(self):
        pass
        
    
    def filter(
        self, 
        x: Union[tf.Tensor, np.ndarray], 
        y: Union[tf.Tensor, np.ndarray], 
        inc_labels: Optional[Union[np.ndarray, list]] = None, 
        excl_labels: Optional[Union[np.ndarray, list]] = None, 
    ) -> Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]:
        """
        Filters dataset by labels.

        Args:
            inc_labels: labels to include. Defaults to None.
            excl_labels: labels to exclude. Defaults to None.

        Returns:
            filtered datasets
        """
        assert (inc_labels is not None) or (excl_labels is not None), "specify labels to filter with"
        labels = np.unique(y)
        split = []
        for l in labels:
            if (inc_labels is None) and (l not in excl_labels):
                split.append(l)
            elif (excl_labels is None) and (l in inc_labels):
                split.append(l)
        labels = np.array([1 if y in split else 0 for y in y])
        x_id = x[np.where(labels)]
        y_id = y[np.where(labels)]

        x_ood = x[np.where(1 - labels)]
        y_ood = y[np.where(1 - labels)]

        return  (x_id, y_id), (x_ood, y_ood)


    def merge(
        self, 
        x_id: Union[tf.Tensor, np.ndarray], 
        x_ood: Union[tf.Tensor, np.ndarray], 
        shuffle: Optional[bool] = False
    )-> Tuple[Union[tf.Tensor, np.ndarray], Union[tf.Tensor, np.ndarray]]:
        """
        Merges two datasets

        Args:
            x_id: ID inputs
            x_ood: OOD inputs (often not used in )

        Returns:
            x: merge dataset
            labels: 1 if ood else 0
        """

        x = np.concatenate([x_id, x_ood])
        labels = np.concatenate([np.zeros(x_id.shape[0]), np.ones(x_ood.shape[0])])

        if shuffle:
            shuffled_inds = np.random.shuffle([i for i in range(x.shape[0])])
            x = x[shuffled_inds]
            labels = labels[shuffled_inds]
        return x, labels

    @staticmethod
    def load(
        key: str,
        **kwargs
    ) -> Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]:
        """
        _summary_

        Args:
            key: _description_
        """
        assert hasattr(tf.keras.datasets, key), f"{key} not available with keras.datasets"
        (x_train, y_train), (x_test, y_test) = keras_dataset_load(key, **kwargs)
        return (x_train, y_train), (x_test, y_test)

    def filter_tfds(
        self, 
        x: Union[tf.Tensor, np.ndarray], 
        inc_labels: Optional[Union[np.ndarray, list]] = None, 
        excl_labels: Optional[Union[np.ndarray, list]] = None, 
    ) -> Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]:
        """
        Filters dataset by labels.

        Args:
            inc_labels: labels to include. Defaults to None.
            excl_labels: labels to exclude. Defaults to None.

        Returns:
            filtered datasets
        """
        assert (inc_labels is not None) or (excl_labels is not None), "specify labels to filter with"
        labels = x.map(lambda x, y: y).unique()
        labels = list(labels.as_numpy_iterator())
        split = []
        for l in labels:
            if (inc_labels is None) and (l not in excl_labels):
                split.append(l)
            elif (excl_labels is None) and (l in inc_labels):
                split.append(l)

        x_id = x.filter(lambda x, y: tf.reduce_any(tf.equal(y, split)))
        x_ood = x.filter(lambda x, y: not tf.reduce_any(tf.equal(y, split)))

        return  x_id, x_ood


    def merge_tfds(
        self, 
        x_id: Union[tf.Tensor, np.ndarray], 
        x_ood: Union[tf.Tensor, np.ndarray], 
        shape: Optional[Tuple[int]] = None,
        shuffle: Optional[bool] = False,
    )-> Tuple[Union[tf.Tensor, np.ndarray], Union[tf.Tensor, np.ndarray]]:
        """
        Merges two datasets

        Args:
            x_id: ID inputs
            x_ood: OOD inputs (often not used in )

        Returns:
            x: merge dataset
            labels: 1 if ood else 0
        """

        if shape is None:
            for x0, y in x_id.take(1):
                shape = x0.shape[:-1]

        def reshape_im(x, y, shape):
            x = tf.image.resize(x, shape)
            return x, y

        x_id = x_id.map(lambda x, y: reshape_im(x, y, shape))
        x_ood = x_ood.map(lambda x, y: reshape_im(x, y, shape))

        def add_label(x, y, label):
            #x.update({'label_ood': label})
            return x, y, label

        x_id = x_id.map(lambda x, y: add_label(x, y, 0))
        x_ood = x_ood.map(lambda x, y: add_label(x, y, 1))
        x = x_id.concatenate(x_ood)

        if shuffle:
            x = x.shuffle(buffer_size = x.cardinality())
        return x


    def get_ood_labels(
        self,
        dataset: Union[tf.Tensor, np.ndarray],
    ):
        
        for x in dataset.take(1):
            if isinstance(x, tuple):
                if len(x) != 3:
                    print("No ood labels to get")
                    return
            else:
                print("No ood labels to get")
                return

        labels = dataset.map(lambda x, y, z: z)
        labels = list(labels.as_numpy_iterator())
        return np.array(labels)

    def load_tfds(
        self,
        dataset_name: Union[str, Tuple],
        preprocess: bool = False,
        preprocessing_fun: Optional[Callable] = None,
        as_numpy: bool = False,
        **kwargs
    ) -> tf.data.Dataset:
        """
        _summary_

        Args:
            key: _description_

        Returns:
            _description_
        """

        dataset = tfds.load(dataset_name, as_supervised=True, **kwargs)
        if preprocess:
            assert preprocessing_fun is not None, "Please specify a preprocessing function"
            for key in dataset.keys():
                dataset[key] = dataset[key].map(
                    lambda x, y: (preprocessing_fun(x), y)
                )
        if as_numpy:
            np_datasets = [self.convert_to_numpy(dataset[key]) for key in dataset.keys()]
            return np_datasets
        else:
            return dataset

    @staticmethod
    def convert_to_numpy(
        dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray]:
        """
        _summary_

        Args:
            dataset: _description_

        Returns:
            _description_
        """

        length = dataset_nb_columns(dataset)
        
        if length == 2:
            x = dataset.map(lambda x, y: x) 
            y = dataset.map(lambda x, y: y) 
            x = np.array(list(x.as_numpy_iterator()))
            y = np.array(list(y.as_numpy_iterator()))
            return x, y

        elif length == 3:
            x = dataset.map(lambda x, y, z: x) 
            y = dataset.map(lambda x, y, z: y) 
            z = dataset.map(lambda x, y, z: z) 
            x = np.array(list(x.as_numpy_iterator()))
            y = np.array(list(y.as_numpy_iterator()))
            z = np.array(list(z.as_numpy_iterator()))
            return x, y, z

        else:
            x = np.array(list(x.as_numpy_iterator()))
            return x


