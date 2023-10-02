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
import tensorflow as tf

from ..types import Optional
from ..types import Tuple


class TFRandomPatchPermutation:
    def __init__(self, patch_size: Tuple[int] = (8, 8)):
        """Randomly permute the patches of an image. This transformation is used in NMD
        paper to artificially craft OOD data from ID images.

        Source (NMD paper):
            "Neural Mean Discrepancy for Efficient Out-of-Distribution Detection"
            [link](https://arxiv.org/pdf/2104.11408.pdf)

        Args:
            patch_size (Tuple[int], optional): Patch dimensions (h, w), should be
                divisors of the image dimensions (H, W). Defaults to (8, 8).
        """
        self.patch_size = patch_size

    def __call__(self, tensor: tf.Tensor, seed: Optional[int] = None):
        """Apply random patch permutation.

        Args:
            tensor (tf.Tensor): Tensor of shape [H, W, C]
            seed (Optinal[int]): Seed number to set for the permutation if not None.

        Returns:
            tf.Tensor: Transformed tensor.
        """
        h, w = self.patch_size
        H, W, C = tensor.shape
        tensor_ = tensor

        # raise warning if patch dimensions are not divisors of image dimensions
        if H % h != 0:
            print(
                f"Warning! Patch height ({h}) should be a divisor of the image height"
                + f" ({H}). Zero padding will be added to get the correct output shape."
            )
            tensor_ = tensor[: -(H % h)]
        if W % w != 0:
            print(
                f"Warning! Patch width ({w}) should be a divisor of the image width"
                + f" ({W}). Zero padding will be added to get the correct output shape."
            )
            tensor_ = tensor_[:, : -(W % w)]

        # === patch permutation ===
        # divide the batch of images into non-overlapping patches
        # => [num_patches, h * w, C]
        u = tf.transpose(
            tf.reshape(tensor_, (H // h, h, W // w, w, C)), (0, 2, 1, 3, 4)
        )
        u = tf.reshape(u, (-1, h * w, C))

        # permute the patches of each image in the batch
        # => [num_patches, h * w, C]
        # Note: we use numpy rng for deterministic index shuffling because
        #       `tf.stateless_shuffle` is still experimental
        g = np.random.default_rng(seed=seed)
        indices = np.arange(u.shape[0])
        g.shuffle(indices)
        pu = tf.gather(u, indices)

        # fold the permuted patches back together
        # => [H, W, C]
        f = tf.transpose(tf.reshape(pu, (H // h, W // w, h, w, C)), (0, 2, 1, 3, 4))
        f = tf.reshape(f, tensor_.shape)
        f = tf.pad(f, tf.constant([[0, H % h], [0, W % w], [0, 0]]))
        return f
