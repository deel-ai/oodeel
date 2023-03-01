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

from ..types import Callable
from ..types import Optional
from ..types import Tuple
from ..types import TypeVar
from ..types import Union
from ..utils import dataset_cardinality
from ..utils import dataset_len_elem
from .tf_data_handler import TFDataHandler

OODDataset = TypeVar("OODDataset", bound="OODDadaset")


class OODDataset(object):
    """Class for managing loading and processing of datasets that are to be used for
    OOD detection. The class encapsulates a dataset like object augmented with OOD
    related inforamtion, and then returrns a dataset like object that is suited for
    scoring or training with the .prepare method.

    Args:
        dataset_id (Union[tf.data.Dataset, tuple, dict, str]): The dataset to load.
            Can be loaded from tensorflow_datasets catalog when the str mathches one of
            the datasets. Defaults to Union[tf.data.Dataset, tuple, dict, str].
        from_directory (bool, optional): If the dataset has to be loaded from directory,
            when dataset_id is str. Defaults to False.
        is_out (bool, optional): If the dataset has to be considered out-of-distribution
            or not. Defaults to False.
        in_value (int, optional): The label to assign to in-distribution samples.
            Defaults to 0.
        out_value (int, optional): The label to assign to out-of-distribution samples.
            Defaults to 1.
        backend (str, optional): Wether the dataset is to be used for tensorflow
             or pytorch models. Defaults to "tensorflow".
        split (str, optional): Split to use ('test' or 'train') When the dataset is
            loaded from tensorflow_dataset. Defaults to None.
        load_kwargs (dict, optional): Additional loading kwargs when loading from
            tensorflow_datasets catalog. Defaults to {}.
    """

    def __init__(
        self,
        dataset_id: Union[tf.data.Dataset, tuple, dict, str],
        from_directory: bool = False,
        in_value: int = 0,
        out_value: int = 1,
        backend: str = "tensorflow",
        split: str = None,
        load_kwargs: dict = {},
    ):
        self.in_value = in_value
        self.out_value = out_value
        self.backend = backend

        # OOD labels are kept as attribute to avoid iterating over the dataset
        self.length = None

        # Set the load parameters for tfds
        if load_kwargs is None:
            load_kwargs = {}
        load_kwargs["as_supervised"] = False
        load_kwargs["split"] = split
        self.load_params = load_kwargs

        # Set the channel order depending on the backend
        if self.backend in ["torch", "pytorch"]:
            tf.config.set_visible_devices([], "GPU")
            self.channel_order = "channels_first"
        else:
            self.channel_order = "channels_last"

        # Load the data handler
        self.data_handler = TFDataHandler()

        # Load the dataset depending on the type of dataset_id
        if isinstance(dataset_id, tf.data.Dataset):
            self.data = self.data_handler.load_tf_ds(dataset_id)

        elif isinstance(dataset_id, (np.ndarray, tuple, dict)):
            self.data = self.data_handler.load_tf_ds_from_numpy(dataset_id)

        elif isinstance(dataset_id, str):
            if from_directory:
                assert os.path.exists(dataset_id), f"Path {dataset_id} does not exist"
                print(f"Loading from directory {dataset_id}")
                # TODO
            else:
                self.data, infos = self.data_handler.load_tf_ds_from_tfds(
                    dataset_id, load_kwargs
                )
                self.length = infos.splits[split].num_examples

        # Get the length of the elements in the dataset
        if self.has_ood_label:
            self.len_elem = dataset_len_elem(self.data) - 1
        else:
            self.len_elem = dataset_len_elem(self.data)

        # Get the key of the tensor to feed the model with
        self.input_key = self.data_handler.get_ds_feature_keys(self.data)[0]

    def __len__(self):
        """get the length of the dataset.

        Returns:
            int: length of the dataset
        """
        if self.length is None:
            self.length = dataset_cardinality(self.data)
        return self.length

    @property
    def has_ood_label(self):
        """Check if the dataset has an out-of-distribution label.

        Returns:
            bool: True if the dataset has an out-of-distribution label.
        """
        return self.data_handler.has_key(self.data, "ood_label")

    def get_ood_labels(
        self,
    ) -> np.ndarray:
        """Get labels from a merged dataset built with ID and OOD data.

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset to get labels from

        Returns:
            np.ndarray: array of labels
        """
        assert self.data_handler.has_key(
            self.data, "ood_label"
        ), "The data has no ood_labels"
        labels = self.data.map(lambda x: x["ood_label"])
        labels = list(labels.as_numpy_iterator())
        return np.array(labels)

    def add_out_data(
        self,
        out_dataset: Union[OODDataset, tf.data.Dataset],
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
    ) -> OODDataset:
        """Concatenate two OODDatasets. Useful for scoring on multiple datasets, or
        training with added out-of-distribution data.

        Args:
            out_dataset (Union[OODDataset, tf.data.Dataset]): dataset of
                out-of-distribution data
            out_as_in (bool, optional): To consider out_dataset as ood or not.
                Defaults to False.
            resize (Optional[bool], optional):toggles if input tensors of the
                datasets have to be resized to have the same shape. Defaults to False.
            shape (Optional[Tuple[int]], optional):shape to use for resizing input
                tensors. If None, the tensors are resized with the shape of the
                in_dataset input tensors. Defaults to None.

        Returns:
            OODDataset: a Dataset object with the concatenated data
        """

        # Creating an OODDataset object from out_dataset if necessary and make sure
        # the two OODDatasets have compatible parameters
        if isinstance(out_dataset, OODDataset):
            out_dataset = out_dataset.data
        else:
            out_dataset = OODDataset(out_dataset).data

        # Assign the correct ood_label to self.data, depending on out_as_in
        self.data = self.data_handler.assign_feature_value(
            self.data, "ood_label", self.in_value
        )
        out_dataset = self.data_handler.assign_feature_value(
            out_dataset, "ood_label", self.out_value
        )

        # Merge the two underlying tf.data.Datasets
        data = self.data_handler.merge(
            self.data,
            out_dataset,
            resize=resize,
            shape=shape,
            channel_order=self.channel_order,
        )

        # Create a new OODDataset from the merged tf.data.Dataset
        output_ds = OODDataset(
            dataset_id=data,
            in_value=self.in_value,
            out_value=self.out_value,
            backend=self.backend,
        )

        return output_ds

    def assign_ood_labels_by_class(
        self,
        in_labels: Optional[Union[np.ndarray, list]] = None,
        out_labels: Optional[Union[np.ndarray, list]] = None,
    ) -> Optional[Tuple[OODDataset]]:
        """Filter the dataset by assigning ood labels depending on labels
        value (typically, class id).

        Args:
            in_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as in-distribution. Defaults to None.
            ood_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as out-of-distribution. Defaults to None.
            return_filtered_ds (bool, optional): To return the filtered
                datasets (in-distribution and out-of-distribution). Defaults to False.

        Returns:
            Optional[Tuple[OODDataset]]: Tuple of in-distribution and
                out-of-distribution OODDatasets
        """
        # Make sure the dataset has labels
        assert (in_labels is not None) or (
            out_labels is not None
        ), "specify labels to filter with"
        assert self.len_elem >= 2, "the dataset has no labels"

        # Filter the dataset depending on in_labels and out_labels given
        if (out_labels is not None) and (in_labels is not None):
            in_data = self.data_handler.filter_by_feature_value(
                self.data, "label", in_labels
            )
            out_data = self.data_handler.filter_by_feature_value(
                self.data, "label", out_labels
            )

        if out_labels is None:
            in_data = self.data_handler.filter_by_feature_value(
                self.data, "label", in_labels
            )
            out_data = self.data_handler.filter_by_feature_value(
                self.data, "label", in_labels, excluded=True
            )

        elif in_labels is None:
            in_data = self.data_handler.filter_by_feature_value(
                self.data, "label", out_labels, excluded=True
            )
            out_data = self.data_handler.filter_by_feature_value(
                self.data, "label", out_labels
            )

        # Assign the correct ood_label to the filtered datasets
        in_data = self.data_handler.assign_feature_value(
            in_data, "ood_label", self.in_value
        )

        out_data = self.data_handler.assign_feature_value(
            out_data, "ood_label", self.out_value
        )

        # Return the filtered OODDatasets
        return OODDataset(
            in_data,
            in_value=self.in_value,
            out_value=self.out_value,
            backend=self.backend,
        ), OODDataset(
            out_data,
            in_value=self.in_value,
            out_value=self.out_value,
            backend=self.backend,
        )

    def prepare(
        self,
        batch_size: int = 128,
        preprocess_fn: Callable = None,
        with_ood_labels: bool = False,
        with_labels: bool = True,
        shuffle: bool = False,
        shuffle_buffer_size: int = None,
        augment_fn: Callable = None,
    ) -> tf.data.Dataset:
        """Prepare self.data for scoring or training

        Args:
            batch_size (int, optional): Batch_size of the returned dataset like obejct.
                Defaults to 128.
            preprocess_fn (Callable, optional): Preprocessing function to apply to
                the dataset. Defaults to None.
            with_ood_labels (bool, optional): To return the dataset with ood_labels
                or not. Defaults to True.
            with_labels (bool, optional): To return the dataset with labels or not.
                Defaults to True.
            shuffle (bool, optional): To shuffle the returned dataset or not.
                Defaults to False.
            shuffle_buffer_size (int, optional): Size of the shuffle buffer. If None,
                taken as the number of samples in the dataset. Defaults to None.
            augment_fn (Callable, optional): Augment function to be used (when the
                returned dataset is to be used for taining). Defaults to None.

        Returns:
            tf.data.Dataset: prepared dataset
        """
        # Check if the dataset has at least one of label and ood_label
        assert (
            with_ood_labels or with_labels
        ), "The dataset must have at least one of label and ood_label"

        # Check if the dataset has ood_labels when asked to return with_ood_labels
        if with_ood_labels:
            assert (
                self.has_ood_label
            ), "Please assign ood labels before preparing with ood_labels"

        dataset_to_prepare = self.data

        # Making the dataset channel first if the backend is pytorch
        if self.backend in ["torch", "pytorch"]:
            dataset_to_prepare = self.data_handler.make_channel_first(
                dataset_to_prepare
            )

        # Select the keys to be returned
        if with_ood_labels and with_labels:
            keys = [self.input_key, "label", "ood_label"]
        elif with_ood_labels and not with_labels:
            keys = [self.input_key, "ood_label"]
        else:
            keys = [self.input_key, "label"]

        # Transform the dataset from dict to tuple
        dataset_to_prepare = self.data_handler.dict_to_tuple(dataset_to_prepare, keys)

        # Apply the preprocessing and augmentation functions if necessary
        if preprocess_fn is not None:
            dataset_to_prepare = self.data_handler.map_ds(
                dataset_to_prepare, preprocess_fn
            )

        if augment_fn is not None:
            dataset_to_prepare = self.data_handler.map_ds(
                dataset_to_prepare, augment_fn
            )

        # Set the shuffle buffer size if necessary
        if shuffle:
            shuffle_buffer_size = (
                len(self) if shuffle_buffer_size is None else shuffle_buffer_size
            )

        # Prepare the dataset for training or scoring
        dataset = self.data_handler.prepare_for_training(
            dataset_to_prepare, batch_size, shuffle_buffer_size
        )

        return dataset
