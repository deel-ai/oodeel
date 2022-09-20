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
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def filter(self, inc_labels=None, excl_labels=None):
        """
        filter dataset by labels

        Args:
            inc_labels: labels to include. Defaults to None.
            excl_labels: labels to exclude. Defaults to None.

        Returns:
            filtered dataset
        """
        assert (inc_labels is not None) or (excl_labels is not None), "specify labels to filter with"
        labels = np.unique(self.y)
        split = []
        for l in labels:
            if (inc_labels is None) and (l not in excl_labels):
                split.append(l)
            elif (excl_labels is None) and (l in inc_labels):
                split.append(l)
        inc_indices = [1 if y in split else 0 for y in self.y]
        x_filter = self.x[np.where(inc_indices)]
        y_filter = self.y[np.where(inc_indices)]
        return x_filter, y_filter