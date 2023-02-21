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
from ..utils import dataset_len_elem

TOODDadaset = TypeVar("TOODDadaset", bound="OODDadaset")


class OODDataset(object):
    # TODO From Generator when reading from files ?
    # TODO Penser à la doc
    # TODO Usage of "ood" is confusing. Can denote the task and a dataset.
    # As a result, ood_label can be ood or id
    def __init__(
        self,
        dataset_id=Union[tf.data.Dataset, tuple, str],
        is_id: bool = None,
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
        self.is_id = is_id

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

        self.len_elem = dataset_len_elem(self.data)
        self.ood_labeled_data = None

        if self.is_id is not None:
            if self.is_id:
                self.assign_ood_label(self.id_value)
            else:
                self.assign_ood_label(self.ood_value)
            self.ood_labels = self.get_ood_labels()

    def has_ood_labels(self):
        return 0 if self.ood_labeled_data is None else 1

    def assign_ood_label(self, ood_label: int):
        """Assign an ood_label to a dataset.

        Args:
            ood_label (int): ood_label to assign
        """
        if self.ood_labeled_data is not None:
            print(
                "Found an existing ood labeled dataset, replacing"
                f" its labels with {ood_label}"
            )

        def assign_ood_label_to_elem(elem):
            elem["ood_label"] = ood_label
            return elem

        self.ood_labeled_data = self.data.map(assign_ood_label_to_elem)

        self.ood_labels = self.get_ood_labels()

    def get_ood_labels(
        self,
    ) -> np.ndarray:
        """Get labels from a merged dataset built with ID and OOD data.

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to get labels from

        Returns:
            np.ndarray: array of labels
        """

        assert self.ood_labeled_data is not None, (
            "OODDataset has no ood labels. Please assign ood labels first "
            "with assign_ood_label()"
        )

        labels = self.ood_labeled_data.map(lambda x: x["ood_label"])
        labels = list(labels.as_numpy_iterator())
        return np.array(labels)

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
            ood_is_id = True
            self.is_id = False
        else:
            ood_is_id = False
            self.is_id = True

        if isinstance(ood_dataset, tf.data.Dataset) or isinstance(ood_dataset, tuple):
            ood_dataset = OODDataset(ood_dataset, backend=self.backend, is_id=ood_is_id)
        else:
            assert (
                self.backend == ood_dataset.backend
            ), "The two datasets have different backends"
            ood_dataset = OODDataset(
                ood_dataset.data, backend=self.backend, is_id=ood_is_id
            )

        if not self.has_ood_labels():
            if ood_is_id:
                self.assign_ood_label(self.ood_value)
            else:
                self.assign_ood_label(self.id_value)

        data = self.merge(self.data, ood_dataset.data, resize=resize, shape=shape)
        ood_labeled_data = self.merge(
            self.ood_labeled_data,
            ood_dataset.ood_labeled_data,
            resize=resize,
            shape=shape,
        )
        output_ds = OODDataset(
            dataset_id=data,
            id_value=self.id_value,
            ood_value=self.ood_value,
            backend=self.backend,
        )
        output_ds.ood_labeled_data = ood_labeled_data
        output_ds.ood_labels = output_ds.get_ood_labels()
        return output_ds

    def assign_ood_labels_by_class(
        self,
        id_labels: Optional[Union[np.ndarray, list]] = None,
        ood_labels: Optional[Union[np.ndarray, list]] = None,
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

        if (id_labels is not None) and (ood_labels is not None):
            if len(self.data.element_spec["label"].shape) > 0:

                def filter_func(elem):
                    label = tf.argmax(elem["label"])
                    return tf.reduce_any(tf.equal(label, id_labels)) + tf.reduce_any(
                        tf.equal(label, ood_labels)
                    )

            else:

                def filter_func(elem):
                    label = elem["label"]
                    return tf.reduce_any(tf.equal(label, id_labels)) + tf.reduce_any(
                        tf.equal(label, ood_labels)
                    )

            self.ood_labeled_data = self.data.filter(filter_func)

        else:
            self.ood_labeled_data = self.data

        def assign_ood_label_to_elem(elem):
            label = elem["label"]
            if ood_labels is None:
                elem["ood_label"] = self.id_value * tf.reduce_any(
                    tf.equal(label, id_labels)
                ) + self.ood_value * (1 - tf.reduce_any(tf.equal(label, id_labels)))
            else:
                elem["ood_label"] = self.id_value * (
                    1 - tf.reduce_any(tf.equal(label, ood_labels))
                ) + self.ood_value * tf.reduce_any(tf.equal(label, ood_labels))
            return elem

        self.ood_labeled_data = self.data.map(assign_ood_label_to_elem)
        self.ood_labels = self.get_ood_labels()
        return self

    def prepare(
        self,
        batch_size: int = 128,
        preprocess_fun: Callable = None,
        with_ood_labels: bool = True,
        with_labels: bool = True,
        training: bool = False,
        input_labels: bool = None,
        buffer_size: int = None,
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

        if preprocess_fun is None:

            def preprocess_fun(x):
                return x

        len_elem = self.len_elem + 1 if with_ood_labels else self.elem

        if self.backend in ["torch", "pytorch"]:
            if len_elem == 1:

                def channel_order(x):
                    return tf.transpose(x, perm=[1, 2, 0])

            else:

                def channel_order(x):
                    tensor = x[0]
                    tensor = tf.transpose(tensor, perm=[1, 2, 0])
                    return tuple([tensor] + list(x[1:]))

        dataset_to_prepare = self.ood_labeled_data if with_ood_labels else self.data

        dataset = dataset_to_prepare.map(
            lambda x: channel_order(preprocess_fun(x)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if training:

            def unroll_dict(elem):
                input_tensor = (
                    tuple([elem[key] for key in input_labels])
                    if isinstance(input_labels, list) or isinstance(input_labels, tuple)
                    else elem[input_labels]
                )
                if with_ood_labels and with_labels:
                    return (
                        input_tensor,
                        elem["label"],
                        elem["ood_label"],
                    )
                elif with_ood_labels and not with_labels:
                    return (
                        input_tensor,
                        elem["ood_label"],
                    )
                return (
                    input_tensor,
                    elem["label"],
                )

            dataset = (
                dataset.map(
                    unroll_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                .map(augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .shuffle(buffer_size=buffer_size)
                .cache()
            )

        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
