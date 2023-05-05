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


def _get_operator(backend):
    if backend == "torch":
        from oodeel.utils.torch_operator import TorchOperator

        return TorchOperator()
    elif backend == "tensorflow":
        from oodeel.utils.tf_operator import TFOperator

        return TFOperator()
    else:
        raise ValueError(f"backend '{backend}' not supported")


def _generate_random_tensor(shape, backend):
    if backend == "torch":
        import torch

        return torch.rand(*shape)
    elif backend == "tensorflow":
        import tensorflow as tf

        return tf.random.uniform(shape)


def _generate_tensor_deterministic(shape, dtype, backend):
    if backend == "torch":
        import torch

        dict_types = {"float32": torch.float32, "int64": torch.int64}
        return torch.arange(np.prod(shape), dtype=dict_types[dtype]).view(*shape)
    elif backend == "tensorflow":
        import tensorflow as tf

        return tf.reshape(tf.range(tf.reduce_prod(shape), dtype=dtype), shape)


def check_common_operators(backend):
    operator = _get_operator(backend)

    input_shape = (25, 12, 6)
    x = _generate_random_tensor(input_shape, backend)
    z = _generate_tensor_deterministic((2, 2, 2), dtype="float32", backend=backend)
    to_one_hot = (
        _generate_tensor_deterministic((3, 2), dtype="int64", backend=backend) % 3
    )

    # Softmax
    softmax_z = operator.softmax(z)
    assert softmax_z.shape == (2, 2, 2)
    np.testing.assert_almost_equal(softmax_z[0, 0, 0], 0.26894143)
    np.testing.assert_almost_equal(softmax_z[0, 0, 1], 0.73105854)

    # Argmax
    assert operator.argmax(z) == 7
    assert operator.argmax(z, dim=1).shape == (2, 2)
    np.testing.assert_array_equal(operator.argmax(z, dim=2), [[1, 1], [1, 1]])

    # Max
    assert operator.max(2 * z) == 14
    assert operator.max(z, dim=1).shape == (2, 2)
    np.testing.assert_array_equal(operator.max(z, dim=2), [[1, 3], [5, 7]])

    # One-hot
    num_classes = 5
    one_hot_tensor = operator.one_hot(to_one_hot, num_classes)
    assert one_hot_tensor.shape == (3, 2, 5)
    assert one_hot_tensor[1, 0, 2] == 1
    np.testing.assert_array_equal(one_hot_tensor[..., 3], 0)

    # Sign
    sign_x = operator.sign(x + 0.01)  # To ensure that all elements are positive
    assert sign_x.shape == input_shape
    np.testing.assert_array_equal(sign_x, 1)
    np.testing.assert_array_equal(operator.sign(x - x), 0)

    # Norm
    assert operator.norm(x, dim=1).shape == (25, 6)
    np.testing.assert_almost_equal(operator.norm(z), 11.832159, decimal=4)
    np.testing.assert_array_almost_equal(
        operator.norm(z, dim=1), [[2.0, 3.1622], [7.2111, 8.6023]], decimal=4
    )

    # Matmul
    x1 = _generate_random_tensor((10, 16), backend)
    x2 = _generate_random_tensor((16, 3), backend)
    assert operator.matmul(x1, x2).shape == (10, 3)

    # Stack
    assert operator.stack([x, x], dim=0).shape == (2, 25, 12, 6)
    assert operator.stack([x, x], dim=1).shape == (25, 2, 12, 6)

    # Cat
    assert operator.cat([x, x], dim=0).shape == (50, 12, 6)
    assert operator.cat([x, x], dim=1).shape == (25, 24, 6)

    # Mean
    assert operator.mean(z, dim=None) == 3.5
    assert operator.mean(x, dim=0).shape == (12, 6)
    assert operator.mean(x, dim=1).shape == (25, 6)

    # Flatten (to 2D tensor)
    assert operator.flatten(x).shape == (25, 12 * 6)

    # Transpose
    assert operator.transpose(x[0]).shape == (6, 12)

    # Diag
    assert operator.diag(x[0]).shape == (6,)

    # Reshape
    assert operator.reshape(x, (30, 2, 30)).shape == (30, 2, 30)

    # Equal
    ind = operator.equal(z, 0)
    assert ind.shape == (2, 2, 2)
    assert z[ind].shape == (1,)

    # Pinv
    assert operator.pinv(x[0]).shape == (6, 12)
