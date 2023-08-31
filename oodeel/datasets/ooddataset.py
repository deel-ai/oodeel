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
import numpy as np

from ..types import Callable
from ..types import DatasetType
from ..types import Optional
from ..types import Tuple
from ..types import Union


class OODDataset(object):
    """Class for managing loading and processing of datasets that are to be used for
    OOD detection. The class encapsulates a dataset like object augmented with OOD
    related information, and then returns a dataset like object that is suited for
    scoring or training with the .prepare method.

    Args:
        dataset_id (Union[DatasetType, tuple, dict, str]): The dataset to load.
            Can be loaded from tensorflow or torch datasets catalog when the str matches
            one of the datasets. Defaults to Union[DatasetType, tuple, dict, str].
        backend (str, optional): Whether the dataset is to be used for tensorflow
             or torch models. Defaults to "tensorflow". Alternative: "torch".
        keys (list, optional): keys to use for dataset elems. Default to None
        load_kwargs (dict, optional): Additional loading kwargs when loading from
            tensorflow_datasets catalog. Defaults to {}.
        load_from_tensorflow_datasets (bool, optional): In the case where if the backend
            is torch but the user still wants to import from tensorflow_datasets
            catalog. In that case, tf.Tensor will not be loaded in VRAM and converted as
            torch.Tensors on the fly. Defaults to False.
        input_key (str, optional): The key of the element/item to consider as the
            model input tensor. If None, taken as the first key. Defaults to None.
    """

    def __init__(
        self,
        dataset_id: Union[DatasetType, tuple, dict, str],
        backend: str = "tensorflow",
        keys: Optional[list] = None,
        load_kwargs: dict = {},
        load_from_tensorflow_datasets: bool = False,
        input_key: Optional[str] = None,
    ):
        self.backend = backend
        self.load_from_tensorflow_datasets = load_from_tensorflow_datasets

        # The length of the dataset is kept as attribute to avoid redundant
        # iterations over self.data
        self.length = None

        # Set the load parameters for tfds / torchvision
        if backend == "tensorflow":
            load_kwargs["as_supervised"] = False
        # Set the channel order depending on the backend
        if self.backend == "torch":
            if load_from_tensorflow_datasets:
                from .tf_data_handler import TFDataHandler
                import tensorflow as tf

                tf.config.set_visible_devices([], "GPU")
                self._data_handler = TFDataHandler()
                load_kwargs["as_supervised"] = False
            else:
                from .torch_data_handler import TorchDataHandler

                self._data_handler = TorchDataHandler()
            self.channel_order = "channels_first"
        else:
            from .tf_data_handler import TFDataHandler

            self._data_handler = TFDataHandler()
            self.channel_order = "channels_last"

        self.load_params = load_kwargs
        # Load the dataset depending on the type of dataset_id
        self.data = self._data_handler.load_dataset(dataset_id, keys, load_kwargs)

        # Get the length of the elements/items in the dataset
        self.len_item = self._data_handler.get_item_length(self.data)
        if self.has_ood_label:
            self.len_item -= 1

        # Get the key of the tensor to feed the model with
        if input_key is None:
            self.input_key = self._data_handler.get_ds_feature_keys(self.data)[0]
        else:
            self.input_key = input_key

    def __len__(self) -> int:
        """get the length of the dataset.

        Returns:
            int: length of the dataset
        """
        if self.length is None:
            self.length = self._data_handler.get_dataset_length(self.data)
        return self.length

    @property
    def has_ood_label(self) -> bool:
        """Check if the dataset has an out-of-distribution label.

        Returns:
            bool: True if data handler has a "ood_label" feature key.
        """
        return self._data_handler.has_feature_key(self.data, "ood_label")

    def get_ood_labels(
        self,
    ) -> np.ndarray:
        """Get ood_labels from self.data if any

        Returns:
            np.ndarray: array of labels
        """
        assert self._data_handler.has_feature_key(
            self.data, "ood_label"
        ), "The data has no ood_labels"
        labels = self._data_handler.get_feature_from_ds(self.data, "ood_label")
        return labels

    def add_out_data(
        self,
        out_dataset: Union["OODDataset", DatasetType],
        in_value: int = 0,
        out_value: int = 1,
        resize: Optional[bool] = False,
        shape: Optional[Tuple[int]] = None,
    ) -> "OODDataset":
        """Concatenate two OODDatasets. Useful for scoring on multiple datasets, or
        training with added out-of-distribution data.

        Args:
            out_dataset (Union[OODDataset, DatasetType]): dataset of
                out-of-distribution data
            in_value (int): ood label value for in-distribution data. Defaults to 0
            out_value (int): ood label value for out-of-distribution data. Defaults to 1
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
        if isinstance(out_dataset, type(self)):
            out_dataset = out_dataset.data
        else:
            out_dataset = OODDataset(out_dataset, backend=self.backend).data

        # Assign the correct ood_label to self.data, depending on out_as_in
        self.data = self._data_handler.assign_feature_value(
            self.data, "ood_label", in_value
        )
        out_dataset = self._data_handler.assign_feature_value(
            out_dataset, "ood_label", out_value
        )

        # Merge the two underlying Datasets
        merge_kwargs = (
            {"channel_order": self.channel_order}
            if self.backend == "tensorflow"
            else {}
        )
        data = self._data_handler.merge(
            self.data,
            out_dataset,
            resize=resize,
            shape=shape,
            **merge_kwargs,
        )

        # Create a new OODDataset from the merged Dataset
        output_ds = OODDataset(
            dataset_id=data,
            backend=self.backend,
        )

        return output_ds

    def split_by_class(
        self,
        in_labels: Optional[Union[np.ndarray, list]] = None,
        out_labels: Optional[Union[np.ndarray, list]] = None,
    ) -> Optional[Tuple["OODDataset"]]:
        """Filter the dataset by assigning ood labels depending on labels
        value (typically, class id).

        Args:
            in_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as in-distribution. Defaults to None.
            out_labels (Optional[Union[np.ndarray, list]], optional): set of labels
                to be considered as out-of-distribution. Defaults to None.

        Returns:
            Optional[Tuple[OODDataset]]: Tuple of in-distribution and
                out-of-distribution OODDatasets
        """
        # Make sure the dataset has labels
        assert (in_labels is not None) or (
            out_labels is not None
        ), "specify labels to filter with"
        assert self.len_item >= 2, "the dataset has no labels"

        # Filter the dataset depending on in_labels and out_labels given
        if (out_labels is not None) and (in_labels is not None):
            in_data = self._data_handler.filter_by_feature_value(
                self.data, "label", in_labels
            )
            out_data = self._data_handler.filter_by_feature_value(
                self.data, "label", out_labels
            )

        if out_labels is None:
            in_data = self._data_handler.filter_by_feature_value(
                self.data, "label", in_labels
            )
            out_data = self._data_handler.filter_by_feature_value(
                self.data, "label", in_labels, excluded=True
            )

        elif in_labels is None:
            in_data = self._data_handler.filter_by_feature_value(
                self.data, "label", out_labels, excluded=True
            )
            out_data = self._data_handler.filter_by_feature_value(
                self.data, "label", out_labels
            )

        # Return the filtered OODDatasets
        return (
            OODDataset(in_data, backend=self.backend),
            OODDataset(out_data, backend=self.backend),
        )

    def prepare(
        self,
        batch_size: int = 128,
        preprocess_fn: Optional[Callable] = None,
        augment_fn: Optional[Callable] = None,
        with_ood_labels: bool = False,
        with_labels: bool = True,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
    ) -> DatasetType:
        """Prepare self.data for scoring or training

        Args:
            batch_size (int, optional): Batch_size of the returned dataset like object.
                Defaults to 128.
            preprocess_fn (Callable, optional): Preprocessing function to apply to
                the dataset. Defaults to None.
            augment_fn (Callable, optional): Augment function to be used (when the
                returned dataset is to be used for training). Defaults to None.
            with_ood_labels (bool, optional): To return the dataset with ood_labels
                or not. Defaults to True.
            with_labels (bool, optional): To return the dataset with labels or not.
                Defaults to True.
            shuffle (bool, optional): To shuffle the returned dataset or not.
                Defaults to False.
            shuffle_buffer_size (int, optional): (TF only) Size of the shuffle buffer.
                If None, taken as the number of samples in the dataset.
                Defaults to None.

        Returns:
            DatasetType: prepared dataset
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

        # Making the dataset channel first if the backend is torch
        if self.backend == "torch" and self.load_from_tensorflow_datasets:
            dataset_to_prepare = self._data_handler.make_channel_first(
                self.input_key, dataset_to_prepare
            )

        # # Select the keys to be returned
        keys = [self.input_key, "label", "ood_label"]
        if not with_labels:
            keys.remove("label")
        if not with_ood_labels:
            keys.remove("ood_label")

        # Prepare the dataset for training or scoring
        dataset = self._data_handler.prepare_for_training(
            dataset=dataset_to_prepare,
            batch_size=batch_size,
            shuffle=shuffle,
            preprocess_fn=preprocess_fn,
            augment_fn=augment_fn,
            output_keys=keys,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        return dataset
