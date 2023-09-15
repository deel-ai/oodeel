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
# -*- encoding: utf-8 -*-
from os import path

from setuptools import find_packages
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "faiss_cpu",
    "numpy",
    "scikit_learn",
    "scipy",
    "setuptools",
    "matplotlib",
    "pandas",
    "seaborn",
    "plotly",
]

tensorflow_requirements = [
    "tensorflow",
    "tensorflow_datasets",
    "tensorflow_probability",
]

torch_requirements = ["timm", "torch", "torchvision"]

dev_requirements = [
    "mypy",
    "ipywidgets",
    "mkdocs-jupyter",
    "mkdocstrings-python",
    "flake8",
    "setuptools",
    "pre-commit",
    "tox",
    "black",
    "ipython",
    "ipykernel",
    "pytest",
    "pylint",
    "mypy",
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings",
    "mknotebooks",
    "bump2version",
    "docsig",
    "no_implicit_optional",
]


docs_requirements = [
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings",
    # only if you want to generate notebooks in documentation website
    "mknotebooks",
    "ipython",
]

setup(
    # Name of the package:
    name="oodeel",
    # Version of the package:
    version="0.2.0",
    # Find the package automatically (include everything):
    packages=find_packages(),
    # Author information:
    author="DEEL Core Team",
    author_email="paul.novello@irt-saintexupery.com",
    # Description of the package:
    description="Simple, compact, and hackable post-hoc deep OOD detection for already"
    + "trained tensorflow or pytorch image classifiers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Plugins entry point
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requirements,
    extras_require={
        "dev": [dev_requirements, tensorflow_requirements, torch_requirements],
        "tensorflow-dev": [dev_requirements, tensorflow_requirements],
        "torch-dev": [dev_requirements, torch_requirements],
        "tensorflow": tensorflow_requirements,
        "torch": torch_requirements,
        "docs": docs_requirements,
    },
)
