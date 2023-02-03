# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Callable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ..utils import dataset_nb_columns
from .tf_load_utils import keras_dataset_load


class DataHandler(object):
    """
    Handles datasets. For now:
    *   filters by labels
    *   merges datasets
    *   loads datasets
    Aims at handling datasets from diverse sources
    """

    def __init__(self):
        pass

    def filter_np(
        self,
        x: np.ndarray,
        y: np.ndarray,
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
        assert (inc_labels is not None) or (
            excl_labels is not None
        ), "specify labels to filter with"
        labels = np.unique(y)
        split = []
        for lbl in labels:
            if (inc_labels is None) and (lbl not in excl_labels):
                split.append(lbl)
            elif (excl_labels is None) and (lbl in inc_labels):
                split.append(lbl)
        labels = np.array([1 if y in split else 0 for y in y])
        x_id = x[np.where(labels)]
        y_id = y[np.where(labels)]

        x_ood = x[np.where(1 - labels)]
        y_ood = y[np.where(1 - labels)]

        return (x_id, y_id), (x_ood, y_ood)

    def merge_np(
        self, x_id: np.ndarray, x_ood: np.ndarray, shuffle: Optional[bool] = False
    ) -> Tuple[Union[tf.Tensor, np.ndarray], Union[tf.Tensor, np.ndarray]]:
        """
        Merges two datasets

        Args:
            x_id: ID inputs
            x_ood: OOD inputs

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
    def load_keras(key: str, **kwargs) -> Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]:
        """
        Loads from keras.datasets

        Args:
            key: dataset name
        """
        assert hasattr(
            tf.keras.datasets, key
        ), f"{key} not available with keras.datasets"
        (x_train, y_train), (x_test, y_test) = keras_dataset_load(key, **kwargs)

        return (x_train, y_train), (x_test, y_test)

    def filter_tfds(
        self,
        x: tf.data.Dataset,
        inc_labels: Optional[Union[np.ndarray, list]] = None,
        excl_labels: Optional[Union[np.ndarray, list]] = None,
    ) -> Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]:
        """
        Filters dataset by labels.

        Args:
            x: tf.data.Dataset to filter
            inc_labels: labels to include. Defaults to None.
            excl_labels: labels to exclude. Defaults to None.

        Returns:
            filtered datasets
        """
        assert (inc_labels is not None) or (
            excl_labels is not None
        ), "specify labels to filter with"
        labels = x.map(lambda x, y: y).unique()
        labels = list(labels.as_numpy_iterator())
        split = []
        for lbl in labels:
            if (inc_labels is None) and (lbl not in excl_labels):
                split.append(lbl)
            elif (excl_labels is None) and (lbl in inc_labels):
                split.append(lbl)

        x_id = x.filter(lambda x, y: tf.reduce_any(tf.equal(y, split)))
        x_ood = x.filter(lambda x, y: not tf.reduce_any(tf.equal(y, split)))

        return x_id, x_ood

    def merge_tfds(
        self,
        x_id: tf.data.Dataset,
        x_ood: tf.data.Dataset,
        shape: Optional[Tuple[int]] = None,
        shuffle: Optional[bool] = False,
    ) -> tf.data.Dataset:
        """
        Merges two tf.data.Datasets

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
            # x.update({'label_ood': label})
            return x, y, label

        x_id = x_id.map(lambda x, y: add_label(x, y, 0))
        x_ood = x_ood.map(lambda x, y: add_label(x, y, 1))
        x = x_id.concatenate(x_ood)

        if shuffle:
            x = x.shuffle(buffer_size=x.cardinality())
        return x

    def get_ood_labels(
        self,
        dataset: tf.data.Dataset,
    ) -> np.ndarray:
        """
        Get labels from a merged dataset built with ID and OOD data.

        Args:
            dataset: tf.data.Dataset to get labels from

        Returns:
            array of labels
        """
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
        **kwargs,
    ) -> tf.data.Dataset:
        """
        Loads dataset from tensorflow-datasets

        Args:
            key: dataset name
            preprocess: preprocess or not
            preprocessing_fun: function used for preprocessing
            as_numpy: returns np.ndarray if True, else tf.data.Dataset

        Returns:
            np.ndarray if True, else tf.data.Dataset
        """

        dataset = tfds.load(dataset_name, as_supervised=True, **kwargs)
        if preprocess:
            assert (
                preprocessing_fun is not None
            ), "Please specify a preprocessing function"
            for key in dataset.keys():
                dataset[key] = dataset[key].map(lambda x, y: (preprocessing_fun(x), y))
        if as_numpy:
            np_datasets = [
                self.convert_to_numpy(dataset[key]) for key in dataset.keys()
            ]
            return np_datasets
        else:
            return dataset

    @staticmethod
    def convert_to_numpy(dataset: tf.data.Dataset) -> Tuple[np.ndarray]:
        """
        converts tf.data.Dataset to numpy

        Args:
            dataset: tf.data.Dataset

        Returns:
            np converted dataset
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
