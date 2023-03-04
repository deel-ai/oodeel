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
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: oodeel_dev_env
#     language: python
#     name: python3
# ---
# +
# %load_ext autoreload
import sys

sys.path.append("../")
import pprint
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow import keras
from oodeel.methods import MLS, DKNN, ODIN
from oodeel.methods import DKNN
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from oodeel.eval.metrics import bench_metrics, get_curve
from oodeel.datasets import OODDataset

from sklearn.metrics import *

import warnings

warnings.filterwarnings("ignore")

pp = pprint.PrettyPrinter()
# -

# ## Two datasets experiment

# +
# %autoreload 2

oods_in = OODDataset("mnist", split="test")
oods_out = OODDataset("fashion_mnist", split="test")
oods_fit = OODDataset("mnist", split="train")


def preprocess_fn(*inputs):
    x = inputs[0] / 255
    return tuple([x] + list(inputs[1:]))


batch_size = 128
ds_in = oods_in.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
ds_out = oods_out.prepare(batch_size=batch_size, preprocess_fn=preprocess_fn)
ds_fit = oods_fit.prepare(
    batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
)

# +

from oodeel.models.training_funs import train_convnet_classifier

try:
    model = tf.keras.models.load_model("../../saved_models/mnist_model")
except OSError:
    train_config = {
        "input_shape": (28, 28, 1),
        "num_classes": 10,
        "is_prepared": True,
        "batch_size": 128,
        "epochs": 5,
        "save_dir": "../saved_models/mnist_model",
        "validation_data": ds_in,  # ds_in is actually the test set of MNIST
    }

    model = train_convnet_classifier(
        ds_fit, **train_config
    )  # ds_fit is actually the train set of MNIST
# -

# ### MLS

# + tags=["MLS two datasets"]

# %autoreload 2


oodmodel = MLS()
oodmodel.fit(model)
scores_in = oodmodel.score(ds_in)
scores_out = oodmodel.score(ds_out)


metrics = bench_metrics(
    (scores_in, scores_out),
    metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
    threshold=-5,
)

pp.pprint(metrics)

# -

# ### DKNN

# +
# %autoreload 2


oodmodel = DKNN()
oodmodel.fit(model, ds_fit.take(10000))
scores_in = oodmodel.score(ds_in.take(1000))
scores_out = oodmodel.score(ds_out.take(1000))


metrics = bench_metrics(
    (scores_in, scores_out),
    metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
    threshold=-5,
)

pp.pprint(metrics)
# -

# ### ODIN

# +

# %autoreload 2
from oodeel.methods import ODIN

oodmodel = ODIN()
oodmodel.fit(model)
scores_in = oodmodel.score(ds_in)
scores_out = oodmodel.score(ds_out)


metrics = bench_metrics(
    (scores_in, scores_out),
    metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
    threshold=-5,
)

pp.pprint(metrics)
# -

# ## Single dataset experiment
#
# (Leave-$k$-classes-out training).
# First, we need to define a training function

# +

# %autoreload 2

oods_test = OODDataset("mnist", split="test")
oods_train = OODDataset("mnist", split="train")

batch_size = 128
inc_labels = [0, 1, 2, 3, 4]
oods_train, _ = oods_train.assign_ood_labels_by_class(in_labels=inc_labels)
oods_in, oods_out = oods_test.assign_ood_labels_by_class(in_labels=inc_labels)


def preprocess_fn(*inputs):
    x = inputs[0] / 255
    return tuple([x] + list(inputs[1:]))


ds_train = oods_train.prepare(
    batch_size=batch_size, preprocess_fn=preprocess_fn, shuffle=True
)
ds_in = oods_in.prepare(
    batch_size=batch_size, with_ood_labels=False, preprocess_fn=preprocess_fn
)
ds_out = oods_out.prepare(
    batch_size=batch_size, with_ood_labels=False, preprocess_fn=preprocess_fn
)


# +
# %autoreload 2
from oodeel.models.training_funs import train_convnet_classifier

train_config = {
    "input_shape": (28, 28, 1),
    "num_classes": len(inc_labels),
    "is_prepared": True,
    "batch_size": 128,
    "epochs": 5,
}

model = train_convnet_classifier(ds_train, **train_config)
# -

# ## MLS

# +
# %autoreload 2

oodmodel = MLS()
oodmodel.fit(model)
scores_in = oodmodel.score(ds_in)
scores_out = oodmodel.score(ds_out)


metrics = bench_metrics(
    (scores_in, scores_out),
    metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
    threshold=-5,
)

pp.pprint(metrics)
# -

# ### DKNN

# +
# %autoreload 2

oodmodel = DKNN()
oodmodel.fit(model, ds_train.take(10000))
scores_in = oodmodel.score(ds_in.take(1000))
scores_out = oodmodel.score(ds_out.take(1000))


metrics = bench_metrics(
    (scores_in, scores_out),
    metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
    threshold=-5,
)

pp.pprint(metrics)
# -

# ### ODIN

# +
# %autoreload 2

# x_test, y_id = data_handler.convert_to_numpy(x_id)

oodmodel = ODIN()
oodmodel.fit(model)
scores_in = oodmodel.score(ds_in)
scores_out = oodmodel.score(ds_out)


metrics = bench_metrics(
    (scores_in, scores_out),
    metrics=["auroc", "fpr95tpr", accuracy_score, roc_auc_score],
    threshold=-5,
)

pp.pprint(metrics)
