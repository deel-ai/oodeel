import tensorflow as tf
import numpy as np


class DataHandler(object):
    """
    Handles datasets (filtering by labels for now).
    Aims at handling datasets from diverse sources

    Args:
        x: inputs
        y: labels
    """
    def __init__(self, data_dir=None):
        self.data_dir=data_dir
        
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

    def merge(self, x_id, x_ood):
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
        return x, labels