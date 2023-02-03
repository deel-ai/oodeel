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
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.append("../")
# os.system("export TFDS_DATA_DIR=/datasets/tensorflow_datasets")

import tensorflow as tf

from oodeel.datasets import DataHandler

# (https://github.com/qubvel/classification_models)
# !pip install image-classifiers==1.0.0b1
from classification_models.tfkeras import Classifiers

# cifar10
data_handler = DataHandler()
normalize = lambda x: x / 255
ds1 = data_handler.load_tfds('cifar10', preprocess=True, preprocessing_fun=normalize)
x_train, x_test = ds1['train'], ds1['test']

# define model
num_classes = 10
input_shape = (32, 32, 3)

ResNet18, _ = Classifiers.get('resnet18')
model = ResNet18(input_shape, classes=num_classes, weights=None)

# train hparams
# (same config as in https://github.com/kuangliu/pytorch-cifar)
batch_size = 128
epochs = 200
lr = 1e-1
# weight_decay = 1e-3
weight_decay = 5e-4

# data aug
padding = 4
image_size = input_shape[0]
target_size = image_size + padding * 2
def _augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels

train_ds = x_train.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# optimizer
decay_steps = int(epochs * len(train_ds) / batch_size)
learning_rate_fn = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(learning_rate_fn, momentum=0.9, weight_decay=weight_decay)

# checkpoint callback
os.makedirs("saved_models", exist_ok=True)
checkpoint_filepath = "saved_models/cifar10"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(train_ds.shuffle(len(train_ds)).batch(batch_size), validation_data=x_test.batch(batch_size), epochs=epochs, callbacks=[model_checkpoint_callback])