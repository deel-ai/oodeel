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
import pytest

from oodeel.datasets import OODDataset
from tests import generate_data
from tests import generate_data_tf


@pytest.mark.parametrize(
    "kwargs_factory, expected_output",
    [
        (lambda: dict(dataset_id="mnist", is_id=True, split="test"), [10000, 2, 3, 0]),
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
                is_id=True,
            ),
            [100, 2, 3, 0],
        ),
        (
            lambda: dict(
                dataset_id=generate_data(
                    x_shape=(32, 32, 3), num_labels=10, samples=100
                ),
                is_id=True,
            ),
            [100, 2, 3, 0],
        ),
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
                is_id=None,
            ),
            [100, 2, 2, None],
        ),
        (
            lambda: dict(
                dataset_id=generate_data(
                    x_shape=(32, 32, 3), num_labels=10, samples=100
                ),
                is_id=None,
            ),
            [100, 2, 2, None],
        ),
    ],
    ids=[
        "Instanciate from tfds",
        "Instanciate from tf dataset as ID",
        "Instanciate from np dataset as ID",
        "Instanciate from tf data without ood labels",
        "Instanciate from np data without ood labels",
    ],
)
def test_instanciate_ood_dataset(kwargs_factory, expected_output):
    """Test the instanciation of OODDataset.
    The instanciation involves using:
    * self.assign_ood_label
    * self.get_ood_labels
    So these methods are considered tested as well
    """
    dataset = OODDataset(**kwargs_factory())

    if dataset.has_ood_labels():
        ood_labels = np.mean(dataset.get_ood_labels())
    else:
        ood_labels = None

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    if kwargs_factory()["is_id"]:
        assert len(dataset.ood_labeled_data.element_spec) == expected_output[2]
    assert ood_labels == expected_output[3]


@pytest.mark.parametrize(
    "kwargs_factory, as_tf_datasets, expected_output",
    [
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
                is_id=True,
            ),
            False,
            [200, 2, 3, 0.5],
        ),
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
            ),
            False,
            [200, 2, 3, 0.5],
        ),
        (
            lambda: dict(
                dataset_id=generate_data_tf(
                    x_shape=(32, 32, 3), num_labels=10, samples=100, as_supervised=False
                ),
                is_id=True,
            ),
            True,
            [200, 2, 3, 0.5],
        ),
    ],
    ids=[
        "Concatenate two Datasets with OOD labels",
        "Concatenate two Datasets without OOD labels",
        "Concatenate a Dataset with OOD labels with a tf.data.Dataset",
    ],
)
def test_concatenate_ood_dataset(kwargs_factory, as_tf_datasets, expected_output):
    """Test the concatenation of OODDataset.
    The concatenation involves using:
    * self.assign_ood_label
    * self.get_ood_labels
    So these methods are considered tested as well
    """

    ds_kwargs = kwargs_factory()
    dataset1 = OODDataset(**ds_kwargs)

    if "is_id" in ds_kwargs.keys():
        ds_kwargs["is_id"] = False

    if as_tf_datasets:
        dataset2 = generate_data(x_shape=(32, 32, 3), num_labels=10, samples=100)
    else:
        dataset2 = OODDataset(**ds_kwargs)

    if dataset1.has_ood_labels():
        dataset = dataset1.concatenate(dataset2)
    else:
        dataset = dataset1.concatenate(dataset2, ood_as_id=True, shape=(23, 23))

    assert len(dataset.data) == expected_output[0]
    assert dataset.len_elem == expected_output[1]
    assert len(dataset.ood_labeled_data.element_spec) == expected_output[2]
    assert np.mean(dataset.get_ood_labels()) == expected_output[3]
