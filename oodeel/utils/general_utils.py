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
from ..types import Any
from ..types import Callable


def is_from(model_or_tensor: Any, framework: str) -> bool:
    """Check whether a model or tensor belongs to a specific framework

    Args:
        model_or_tensor (Any): Neural network or Tensor
        framework (str):  Model or tensor framework ("torch" | "keras" | "tensorflow")

    Returns:
        bool: Whether the model belongs to specified framework or not
    """
    keywords_list = []
    class_parents = list(model_or_tensor.__class__.__mro__)
    for class_id in class_parents:
        class_list = str(class_id).split("'")[1].split(".")
        for keyword in class_list:
            keywords_list.append(keyword)
    return framework in keywords_list


def import_backend_specific_stuff(model: Callable):
    """Get backend specific data handler, operator and feature extractor class.

    Args:
        model (Callable): a model (Keras or PyTorch) used to identify the backend.

    Returns:
        str: backend name
        DataHandler: torch or tf data handler
        Operator: torch or tf operator
        FeatureExtractor: torch or tf feature extractor class
    """
    if is_from(model, "keras"):
        from ..extractor.keras_feature_extractor import KerasFeatureExtractor
        from ..datasets.tf_data_handler import TFDataHandler
        from ..utils import TFOperator

        backend = "tensorflow"
        data_handler = TFDataHandler()
        op = TFOperator()
        FeatureExtractorClass = KerasFeatureExtractor

    elif is_from(model, "torch"):
        from ..extractor.torch_feature_extractor import TorchFeatureExtractor
        from ..datasets.torch_data_handler import TorchDataHandler
        from ..utils import TorchOperator

        backend = "torch"
        data_handler = TorchDataHandler()
        op = TorchOperator(model)
        FeatureExtractorClass = TorchFeatureExtractor

    else:
        raise NotImplementedError()

    return backend, data_handler, op, FeatureExtractorClass
