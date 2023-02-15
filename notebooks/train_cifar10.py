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
#
# For image-classifiers, see:
# (https://github.com/qubvel/classification_models)
# !pip install image-classifiers==1.0.0b1
import os
import sys
import warnings

from oodeel.datasets import DataHandler
from oodeel.models.training_funs import train_keras_app

warnings.filterwarnings("ignore")


sys.path.append("../")

# cifar10
data_handler = DataHandler()

ds1 = data_handler.load_tfds(
    "cifar10", preprocess=True, preprocessing_fun=(lambda x: x / 255)
)
x_train, x_test = ds1["train"], ds1["test"]

os.makedirs("saved_models", exist_ok=True)
checkpoint_filepath = "saved_models/cifar10"

# define model
model = train_keras_app(
    x_train,
    model_name="resnet18",
    epochs=200,
    batch_size=128,
    validation_data=x_test,
    save_dir=checkpoint_filepath,
)
