import tensorflow as tf
import numpy as np
from oodeel.datasets.load_utils import keras_dataset_load

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
        
    
    def filter(self, x, y, inc_labels=None, excl_labels=None, merge=False):
        """
        Filters dataset by labels.

        Args:
            inc_labels: labels to include. Defaults to None.
            excl_labels: labels to exclude. Defaults to None.

        Returns:
            filtered dataset
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


    def merge(self, x_id, x_ood, shuffle=False):
        """
        Merges two datasets

        Args:
            x_id: ID inputs
            x_ood: OOD inputs (often not used in )

        Returns:
            _description_
        """

        x = np.concatenate([x_id, x_ood])
        labels = np.concatenate([np.zeros(x_id.shape[0]), np.ones(x_ood.shape[0])])

        if self.shuffle:
            shuffled_inds = np.random.shuffle([i for i in range(x.shape[0])])
            x = x[shuffled_inds]
            labels = labels[shuffled_inds]
        return x, labels

    @staticmethod
    def load(key):
        if key in ["mnist", "fashion_mnist"]:
            (x_train, y_train), (x_test, y_test) = keras_dataset_load(key)
        else:
            raise NotImplementedError()
        return (x_train, y_train), (x_test, y_test)