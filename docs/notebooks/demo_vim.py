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
import tensorflow as tf

from oodeel.datasets import DataHandler
from oodeel.eval.metrics import bench_metrics
from oodeel.eval.metrics import get_curve
from oodeel.methods.vim import VIM

batch_size = 128

# %% load data


def normalize(x):
    return x / 255


data_handler = DataHandler()
ds1 = data_handler.load_tfds("mnist", preprocess=True, preprocessing_fun=normalize)
ds2 = data_handler.load_tfds(
    "fashion_mnist", preprocess=True, preprocessing_fun=normalize
)

x_id = ds1["test"]
x_ood = ds2["test"]
x_train = ds1["train"]

x_test = data_handler.merge_tfds(x_id, x_ood, shuffle=False)

# %% load model
model = tf.keras.models.load_model("./saved_models/mnist_model")
W = model.layers[-1].get_weights()[0]
print(f"Number of ID classes: {W.shape[1]}")
print(f"Dimension of feature space:  {W.shape[0]}")


# %%  Specify number of principal dimensions
princ_dims = 500
print(
    "OOD method VIM by specifying number of principal "
    f"dimensions in feature space ({princ_dims=})"
)
oodmodel = VIM(princ_dims=princ_dims)
oodmodel.fit(model, x_train.batch(batch_size))

print(f"Number of princpal dimensions: {oodmodel._princ_dim}")

# %% Get scores
scores = oodmodel.score(x_test.batch(batch_size))
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, metrics=["auroc", "fpr95tpr"], threshold=-5, step=1
)

print(f"VIM metrics (princ_dims={princ_dims})")
print(metrics)

# %%
print(
    "OOD method VIM by specifying number of principal"
    f" dimensions in feature space ({princ_dims=})"
)
print(
    "Using the origin described in the original paper for the PCA in feature "
    "space (pca_origin='pseudo')"
)

princ_dims = 500
oodmodel = VIM(princ_dims=princ_dims, pca_origin="pseudo")
oodmodel.fit(model, x_train.batch(batch_size))


# %% Get scores
scores = oodmodel.score(x_test.batch(batch_size))
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, metrics=["auroc", "fpr95tpr"], threshold=-5, step=1
)


print(f"VIM metrics (princ_dims={princ_dims}, pca_origin='pseudo')")
print(f"Number of princpal dimensions: {oodmodel._princ_dim=}")
print(metrics)


# %%
princ_dims = 0.9
print(
    "OOD method VIM by specifying ratio of explained variance "
    f"in feature space ({princ_dims=})"
)
oodmodel = VIM(princ_dims=princ_dims)
oodmodel.fit(model, x_train.batch(batch_size))


# %% Get scores
scores = oodmodel.score(x_test.batch(batch_size))
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, metrics=["auroc", "fpr95tpr"], threshold=-5, step=1
)

print(f"VIM metrics (princ_dims={princ_dims})")
print(f"Number of princpal dimensions: {oodmodel._princ_dim=}")
print(metrics)

# %%
princ_dims = None
print(
    "OOD method VIM by relying on Kneedle to find the number of principal dimensions "
    f"in feature space ({princ_dims=})"
)

oodmodel = VIM(princ_dims=princ_dims)
oodmodel.fit(model, x_train.batch(batch_size))


# %% Get scores
scores = oodmodel.score(x_test.batch(batch_size))
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, metrics=["auroc", "fpr95tpr"], threshold=-5, step=1
)

print(f"VIM metrics (princ_dims={princ_dims})")
print(f"Number of princpal dimensions (found by Kneedle): {oodmodel._princ_dim=}")
print(metrics)
