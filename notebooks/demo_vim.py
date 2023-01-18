# %% Imports
import sys
sys.path.append("./")
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import *

from oodeel.eval.metrics import bench_metrics, get_curve
from oodeel.datasets import DataHandler
from oodeel.methods.vim import VIM


# %% load data
def normalize(x):
    return x/255
    
data_handler = DataHandler()
ds1 = data_handler.load_tfds('mnist', preprocess=True, preprocessing_fun=normalize)
ds2 = data_handler.load_tfds('fashion_mnist', preprocess=True, preprocessing_fun=normalize)

x_id = ds1["test"]
x_ood = ds2["test"]
x_train = ds1["train"]

x_test = data_handler.merge_tfds(x_id, x_ood, shuffle=False)

# %% load model
model = tf.keras.models.load_model("./saved_models/mnist_model")
W= model.layers[-1].get_weights()[0]
print(f"Number of ID classes: {W.shape[1]}")
print(f"Dimension of feature space:  {W.shape[0]}")




# %%  Specify number of principal dimensions
princ_dims=500
print(f"OOD method VIM by specifying number of principal dimensions in feature space ({princ_dims=})")
oodmodel = VIM(princ_dims=princ_dims)
oodmodel.fit(model, x_train)

print(f"Number of princpal dimensions: {oodmodel._princ_dim}")

# %% Get scores 
scores = oodmodel.score(x_test)
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, 
    metrics = ["auroc", "fpr95tpr"], 
    threshold = -5,
    step=1
    )

print(f"VIM metrics (princ_dims={princ_dims})")
print(metrics)

# %% 
print(f"OOD method VIM by specifying number of principal dimensions in feature space ({princ_dims=})")
print("Using the origin described in the original paper for the PCA in feature space (pca_origin='pseudo')")

princ_dims=500
oodmodel = VIM(princ_dims=princ_dims, pca_origin="pseudo")
oodmodel.fit(model, x_train)


# %% Get scores 
scores = oodmodel.score(x_test)
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, 
    metrics = ["auroc", "fpr95tpr"], 
    threshold = -5,
    step=1
    )


print(f"VIM metrics (princ_dims={princ_dims}, pca_origin='pseudo')")
print(f"Number of princpal dimensions: {oodmodel._princ_dim=}")
print(metrics)


# %% 
princ_dims=0.9
print(f"OOD method VIM by specifying ratio of explained variance in feature space ({princ_dims=})")
oodmodel = VIM(princ_dims=princ_dims)
oodmodel.fit(model, x_train)




# %% Get scores 
scores = oodmodel.score(x_test)
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, 
    metrics = ["auroc", "fpr95tpr"], 
    threshold = -5,
    step=1
    )

print(f"VIM metrics (princ_dims={princ_dims})")
print(f"Number of princpal dimensions: {oodmodel._princ_dim=}")
print(metrics)

# %% 
princ_dims=None
print(f"OOD method VIM by relying on Kneedle to find the number of principal dimensions in feature space ({princ_dims=})")

oodmodel = VIM(princ_dims=princ_dims)
oodmodel.fit(model, x_train)




# %% Get scores 
scores = oodmodel.score(x_test)
labels = data_handler.get_ood_labels(x_test)
fpr, tpr = get_curve(scores, labels)

metrics = bench_metrics(
    scores, labels, 
    metrics = ["auroc", "fpr95tpr"], 
    threshold = -5,
    step=1
    )

print(f"VIM metrics (princ_dims={princ_dims})")
print(f"Number of princpal dimensions (found by Kneedle): {oodmodel._princ_dim=}")
print(metrics)