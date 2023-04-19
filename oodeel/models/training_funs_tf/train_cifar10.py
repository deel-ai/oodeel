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
import argparse
import os
import sys
import warnings

from oodeel.datasets import OODDataset
from oodeel.models.training_funs_tf import train_keras_app


warnings.filterwarnings("ignore")


sys.path.append("../")

parser = argparse.ArgumentParser(
    prog="Train CIFAR-10",
    description="Train keras or torch model on CIFAR-10 dataset",
)
parser.add_argument(
    "-e", "--epochs", type=int, default=200, help="Number of epochs (default: 200)"
)
parser.add_argument(
    "-b", "--batch_size", type=int, default=128, help="Batch size (default: 128)"
)
parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default="resnet50",
    help="Model name (default: 'resnet50')",
)
parser.add_argument(
    "-s",
    "--save_dir",
    type=str,
    default="saved_models/cifar10",
    help="Directory where model will be saved" + " (default: 'saved_models/cifar10')",
)
args = parser.parse_args()


if __name__ == "__main__":
    # cifar10
    oods_train = OODDataset("cifar10", split="train")
    oods_test = OODDataset("cifar10", split="test")
    os.makedirs(args.save_dir, exist_ok=True)

    # train model
    model = train_keras_app(
        train_data=oods_train,
        validation_data=oods_test,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )
