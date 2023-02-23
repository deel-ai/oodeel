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
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ..types import Callable
from ..types import Optional
from ..types import Tuple
from ..types import TypeVar
from ..types import Union
from ..utils import dataset_cardinality
from ..utils import dataset_len_elem

TOODDadaset = TypeVar("TOODDadaset", bound="OODDadaset")


class OODDataset(object):
    # TODO Penser à la doc
    # TODO Usage of "ood" is confusing. Can denote the task and a dataset.
    # As a result, ood_label can be ood or id
    # TODO Faire un loader à part ?
    def __init__(
        self,
        dataset_id=Union[tf.data.Dataset, tuple, str],
        is_ood: bool = False,
        id_value: int = 0,
        ood_value: int = 1,
        backend: str = "tf",
        split: str = None,
        load_kwargs: dict = {},
    ):
        self.id_value = id_value
        self.ood_value = ood_value
        self.load_params = load_kwargs
        self.backend = backend
        self.ood_labels = None
        self.is_ood = is_ood

        if self.backend in ["torch", "pytorch"]:
            tf.config.set_visible_devices([], "GPU")
            self.channel_order = "channels_first"
        else:
            self.channel_order = "channels_last"

        if isinstance(dataset_id, tf.data.Dataset):
            assert isinstance(dataset_id.element_spec, dict), (
                "Please provide a dataset with elements as a dict instead of a tuple. "
                "For instance, use tf.data.Dataset.from_tensor_slices({'input': x, "
                "'label': y}) instead of tf.data.Dataset.from_tensor_slices((x, y))"
            )
            self.data = dataset_id

        elif isinstance(dataset_id, np.ndarray):
            dataset_dict = {"input": dataset_id}

        elif isinstance(dataset_id, tuple):
            len_elem = len(dataset_id)
            if len_elem == 2:
                dataset_dict = {"input": dataset_id[0], "label": dataset_id[1]}
            else:
                dataset_dict = {
                    f"input_{i}": dataset_id[i] for i in range(len_elem - 1)
                }
                dataset_dict["label"] = dataset_id[-1]
            print(
                'Loading tf.data.Dataset with elems as dicts, assigning "input_i" key'
                ' to the i-th tuple dimension and "label" key to the last '
                "tuple dimension."
            )
            self.data = tf.data.Dataset.from_tensor_slices(dataset_dict)

        elif isinstance(dataset_id, str):
            if dataset_id in tfds.list_builders():
                print("Loading from tensorflow_datasets")
                if "as_supervised" in load_kwargs.keys():
                    if load_kwargs["as_supervised"]:
                        print(
                            "as_supervised must be False when loading from"
                            " tensorflow datasets. Changing to True."
                        )
                load_kwargs["as_supervised"] = False
                self.data = tfds.load(dataset_id, split=split, **load_kwargs)
                assert isinstance(self.data, tf.data.Dataset), (
                    "Please specify a split for loading from tensorflow_datasets"
                    " (train, test, ...)"
                )
            else:
                assert os.path.exists(dataset_id), f"Path {dataset_id} does not exist"
                print(f"Loading from directory {dataset_id}")
                # TODO
                raise NotImplementedError()

        try:
            self.length = len(self.data)
        except TypeError:
            self.length = None

        if self.has_ood_labels():
            self.len_elem = dataset_len_elem(self.data) - 1
        else:
            self.len_elem = dataset_len_elem(self.data)

        if self.is_ood:
            self.assign_ood_label(self.ood_value)
        elif (not self.is_ood) and (self.is_ood is not None):
            self.assign_ood_label(self.id_value)

        self.input_key = list(self.data.element_spec.keys())[0]

    def has_ood_labels(self):
        return 1 if ("ood_label" in self.data.element_spec.keys()) else 0

    def cardinality(self):
        if self.length is not None:
            return self.length
        else:
            self.length = dataset_cardinality(self.data)
            return self.length

    def assign_ood_label(self, ood_label: int, replace: bool = None):
        """Assign an ood_label to a dataset.

        Args:
            ood_label (int): ood_label to assign
        """
        if not self.has_ood_labels() or replace:

            def assign_ood_label_to_elem(elem):
                elem["ood_label"] = ood_label
                return elem

            self.data = self.data.map(assign_ood_label_to_elem)

        self.ood_labels = np.array([ood_label for i in range(self.cardinality())])

    def get_ood_labels_from_ds(self) -> np.ndarray:
        """Get labels from a merged dataset built with ID and OOD data.

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to get labels from

        Returns:
            np.ndarray: array of labels
        """

        assert self.has_ood_labels(), (
            "The input dataset has no ood labels. Please assign ood labels first "
            "with assign_ood_label()"
        )

        labels = self.data.map(lambda x: x["ood_label"])
        labels = list(labels.as_numpy_iterator())
        self.ood_labels = np.array(labels)
        return self.ood_labels

    def merge(
        self,
        id_dataset: tf.data.Dataset,
        ood_dataset: tf.data.Dataset,
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
    ) -> tf.data.Dataset:
        """Merge two tf.data.Datasets

        Args:
            id_dataset (tf.data.Dataset): dataset of in-distribution data
            ood_dataset (tf.data.Dataset): dataset of out-of-distribution data
                resize (Optional[bool], optional): toggles if input tensors of the
                datasets have to be resized to have the same shape. Defaults to True.
            shape (Optional[Tuple[int]], optional): shape to use for resizing input
                tensors. If None, the tensors are resized with the shape of the
                id_dataset
            input tensors . Defaults to None.

        Returns:
            tf.data.Dataset: merged dataset with ood labels
        """
        len_elem_id = dataset_len_elem(id_dataset)
        len_elem_ood = dataset_len_elem(ood_dataset)
        assert (
            len_elem_id == len_elem_ood
        ), "incompatible dataset elements (different elem dict length)"

        if shape is not None:
            resize = True

        input_key_id = list(id_dataset.element_spec.keys())[0]
        input_key_ood = list(ood_dataset.element_spec.keys())[0]
        shape_id = id_dataset.element_spec[input_key_id].shape
        shape_ood = ood_dataset.element_spec[input_key_ood].shape

        if shape_id != shape_ood:
            resize = True

            if shape is None:
                print(
                    "Resizing the first item of elem (usually the image)",
                    " with the shape of id_dataset",
                )
                if self.channel_order == "channels_first":
                    shape = shape_id[1:]
                else:
                    shape = shape_id[:2]

        if resize:

            def reshape_im_id(elem):
                elem[input_key_id] = tf.image.resize(elem[input_key_id], shape)
                return elem

            def reshape_im_ood(elem):
                elem[input_key_ood] = tf.image.resize(elem[input_key_ood], shape)
                return elem

            id_dataset = id_dataset.map(reshape_im_id)
            ood_dataset = ood_dataset.map(reshape_im_ood)

        merged_dataset = id_dataset.concatenate(ood_dataset)
        return merged_dataset

    def concatenate(
        self,
        ood_dataset: Union[TOODDadaset, tf.data.Dataset],
        ood_as_id: bool = False,
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
    ) -> tf.data.Dataset:
        """Concatenate self with another OODDataset

        Args:
            id_dataset (tf.data.Dataset): dataset of in-distribution data
            ood_dataset (tf.data.Dataset): dataset of out-of-distribution data
                resize (Optional[bool], optional): toggles if input tensors of the
                datasets have to be resized to have the same shape. Defaults to True.
            shape (Optional[Tuple[int]], optional): shape to use for resizing input
                tensors. If None, the tensors are resized with the shape of the
                id_dataset input tensors. Defaults to None.

        Returns:
            Dataset: a Dataset object with the concatenated data
        """

        if ood_as_id:
            if (not self.is_ood) or (self.is_ood is None):
                self.assign_ood_label(self.ood_value)
        else:
            if self.is_ood is None:
                self.assign_ood_label(self.id_value)

        if isinstance(ood_dataset, (tf.data.Dataset, tuple)):
            ood_dataset = OODDataset(
                ood_dataset, backend=self.backend, is_ood=not ood_as_id
            )
        else:
            assert (
                self.backend == ood_dataset.backend
            ), "The two datasets have different backends"
            ood_dataset = OODDataset(
                ood_dataset.data, backend=self.backend, is_ood=not ood_as_id
            )

        data = self.merge(self.data, ood_dataset.data, resize=resize, shape=shape)

        output_ds = OODDataset(
            dataset_id=data,
            is_ood=None,
            id_value=self.id_value,
            ood_value=self.ood_value,
            backend=self.backend,
        )
        output_ds.ood_labels = np.concatenate([self.ood_labels, ood_dataset.ood_labels])
        return output_ds

    def assign_ood_labels_by_class(
        self,
        id_labels: Optional[Union[np.ndarray, list]] = None,
        ood_labels: Optional[Union[np.ndarray, list]] = None,
        return_filtered_ds: bool = False,
    ) -> Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]:
        """Filter the dataset by assigning ood labels depending on labels
        value (typically, class id).

        Args:
            id_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as in-distribution. Defaults to None.
            ood_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as out-of-distribution. Defaults to None.

        Returns:
            Tuple[Tuple[Union[tf.Tensor, np.ndarray]]]: _description_
        """

        assert (id_labels is not None) or (
            ood_labels is not None
        ), "specify labels to filter with"
        assert self.len_elem == 2, "the dataset has no labels"

        if len(self.data.element_spec["label"].shape) > 0:

            def get_label_int(elem):
                return int(tf.argmax(elem["label"]))

        else:

            def get_label_int(elem):
                return elem["label"]

        def add_ood_label(elem, ood_label):
            elem["ood_label"] = ood_label
            return elem

        def filter_func_id(elem):
            label = get_label_int(elem)

            if (id_labels is not None) and (ood_labels is not None):
                return tf.reduce_any(tf.equal(label, id_labels))

            elif ood_labels is None:
                return tf.reduce_any(tf.equal(label, id_labels))

            else:
                return not tf.reduce_any(tf.equal(label, ood_labels))

        def filter_func_ood(elem):
            label = get_label_int(elem)

            if (id_labels is not None) and (ood_labels is not None):
                return tf.reduce_any(tf.equal(label, ood_labels))

            elif ood_labels is None:
                return not tf.reduce_any(tf.equal(label, id_labels))

            else:
                return tf.reduce_any(tf.equal(label, ood_labels))

        id_data = self.data
        ood_data = self.data

        id_data = id_data.filter(filter_func_id)
        id_data = id_data.map(lambda elem: add_ood_label(elem, self.id_value))
        len_id = dataset_cardinality(id_data)

        ood_data = ood_data.filter(filter_func_ood)
        ood_data = ood_data.map(lambda elem: add_ood_label(elem, self.ood_value))
        len_ood = dataset_cardinality(ood_data)

        self.data = id_data.concatenate(ood_data)
        self.ood_labels = np.concatenate(
            [
                np.array([self.id_value for i in range(len_id)]),
                np.array([self.ood_value for i in range(len_ood)]),
            ]
        )
        self.is_ood = None

        if return_filtered_ds:
            return OODDataset(
                id_data,
                id_value=self.id_value,
                ood_value=self.ood_value,
                backend=self.backend,
            ), OODDataset(
                ood_data,
                is_ood=True,
                id_value=self.id_value,
                ood_value=self.ood_value,
                backend=self.backend,
            )

    def prepare(
        self,
        batch_size: int = 128,
        preprocess_fn: Callable = None,
        as_supervised: bool = False,
        with_ood_labels: bool = True,
        with_labels: bool = True,
        shuffle: bool = False,
        shuffle_buffer_size: int = None,
        augment_fn: Callable = None,
    ) -> tf.data.Dataset:
        """prepare self.data and self.ood_labeled_data for scoring

        Args:
            batch_size (int, optional): batch_size for scoring. Defaults to 128.
            preprocess_fun (Callable, optional): preprocessing of data to score.
                Defaults to None.
            training (bool, optional): toggle advanced preparation for training
                (augmentation and shuffling). Defaults to False.
            buffer_size (int, optional): buffer_size for shuffling. Defaults to None.
            augment_fn (Callable, optional): data augmentation function.
                Defaults to None.

        Returns:
            tf.data.Dataset: prepared dataset
        """

        assert (
            with_ood_labels or with_labels
        ), "The dataset must have at least one of label and ood_label"

        if with_ood_labels:
            assert (
                self.has_ood_labels()
            ), "Please assign ood labels before preparing with ood_labels"

        dataset_to_prepare = self.data

        if shuffle or (shuffle_buffer_size is not None):
            shuffle_buffer_size = (
                self.cardinality()
                if shuffle_buffer_size is None
                else shuffle_buffer_size
            )
            dataset_to_prepare = dataset_to_prepare.shuffle(shuffle_buffer_size)

        if self.backend in ["torch", "pytorch"]:

            def channel_order(elem):
                elem[self.input_key] = tf.transpose(
                    elem[self.input_key], perm=[1, 2, 0]
                )
                return elem

            dataset_to_prepare = dataset_to_prepare.map(
                lambda x: channel_order(x),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        if as_supervised:

            def process_dict(elem):
                if with_ood_labels and with_labels:
                    return (
                        elem[self.input_key],
                        elem["label"],
                        elem["ood_label"],
                    )
                elif with_ood_labels and not with_labels:
                    return (
                        elem[self.input_key],
                        elem["ood_label"],
                    )
                return (
                    elem[self.input_key],
                    elem["label"],
                )

        else:

            def process_dict(elem):
                if with_ood_labels and with_labels:
                    return elem
                elif with_ood_labels and not with_labels:
                    elem.pop("label")
                    return elem
                elem.pop("ood_label")
                return elem

        dataset_to_prepare = dataset_to_prepare.map(
            process_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if preprocess_fn is not None:
            dataset_to_prepare = dataset_to_prepare.map(
                preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        if augment_fn is not None:
            dataset_to_prepare = dataset_to_prepare.map(
                augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        dataset = (
            dataset_to_prepare.cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return dataset
