import tensorflow as tf
import numpy as np

def split(x, y, id_inds, ood_inds):
    """
    Splits a dataset for Open Set Recognition benchmarks (e.g. 6 vs 4)

    Parameters
    ----------
    x : input data
    y : corresponding labels
    id_inds : list or array
        classes to consider id
    ood_inds : list or array
        classes to consider ood

    Returns
    -------
    dataset
        split dataset
    """
    x_id = x[np.where(y in id_inds)]
    x_ood = x[np.where(y in ood_inds)]
    y_id = y[np.where(y in id_inds)]
    y_ood = y[np.where(y in ood_inds)]
    return (x_id, y_id), (x_ood, y_ood)