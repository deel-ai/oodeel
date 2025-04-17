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
import inspect
from abc import ABC
from abc import abstractmethod
from typing import get_args

import numpy as np
from tqdm import tqdm

from ..extractor.feature_extractor import FeatureExtractor
from ..types import Callable
from ..types import DatasetType
from ..types import ItemType
from ..types import List
from ..types import Optional
from ..types import TensorType
from ..types import Union
from ..utils import import_backend_specific_stuff


class OODBaseDetector(ABC):
    """OODBaseDetector is an abstract base class for Out-of-Distribution (OOD)
    detection.

    Attributes:
        feature_extractor (FeatureExtractor): The feature extractor instance.
        use_react (bool): Flag to indicate if ReAct method is used.
        use_scale (bool): Flag to indicate if scaling method is used.
        use_ash (bool): Flag to indicate if ASH method is used.
        react_quantile (float): Quantile value for ReAct threshold.
        scale_percentile (float): Percentile value for scaling.
        ash_percentile (float): Percentile value for ASH.
        react_threshold (float): Threshold value for ReAct.
        postproc_fns (List[Callable]): List of post-processing functions.

    Methods:
        __init__: Initializes the OODBaseDetector with specified parameters.
        _score_tensor: Abstract method to compute OOD score for input samples.
        _sanitize_posproc_fns: Sanitizes post-processing functions used at each layer
        output.
        fit: Prepares the detector for scoring by constructing the feature extractor
        and calibrating on ID data.
        _load_feature_extractor: Loads the feature extractor based on the model and
        specified layers.
        _fit_to_dataset: Abstract method to fit the OOD detector to a dataset.
        score: Computes an OOD score for input samples.
        compute_react_threshold: Computes the ReAct threshold using the fit dataset.
        __call__: Convenience wrapper for the score method.
        requires_to_fit_dataset: Property indicating if the detector needs a fit
        dataset.
        requires_internal_features: Property indicating if the detector acts on
        internal model features.
    """

    def __init__(
        self,
        use_react: Optional[bool] = False,
        use_scale: Optional[bool] = False,
        use_ash: Optional[bool] = False,
        react_quantile: Optional[float] = None,
        scale_percentile: Optional[float] = None,
        ash_percentile: Optional[float] = None,
        postproc_fns: Optional[List[Callable]] = None,
    ):
        self.feature_extractor: FeatureExtractor = None
        self.use_react = use_react
        self.use_scale = use_scale
        self.use_ash = use_ash
        self.react_quantile = react_quantile
        self.scale_percentile = scale_percentile
        self.ash_percentile = ash_percentile
        self.react_threshold = None
        self.postproc_fns = self._sanitize_posproc_fns(postproc_fns)

        if use_scale and use_react:
            raise ValueError("Cannot use both ReAct and scale at the same time")
        if use_scale and use_ash:
            raise ValueError("Cannot use both ASH and scale at the same time")
        if use_ash and use_react:
            raise ValueError("Cannot use both ReAct and ASH at the same time")

    @abstractmethod
    def _score_tensor(self, inputs: TensorType) -> np.ndarray:
        """Computes an OOD score for input samples "inputs".

        Method to override with child classes.

        Args:
            inputs (TensorType): tensor to score
        Returns:
            Tuple[TensorType]: OOD scores, predicted logits
        """
        raise NotImplementedError()

    def _sanitize_posproc_fns(
        self,
        postproc_fns: Union[List[Callable], None],
    ) -> List[Callable]:
        """Sanitize postproc fns used at each layer output of the feature extractor.

        Args:
            postproc_fns (Optional[List[Callable]], optional): List of postproc
                functions, one per output layer. Defaults to None.

        Returns:
            List[Callable]: Sanitized postproc_fns list
        """
        if postproc_fns is not None:
            assert len(postproc_fns) == len(
                self.output_layers_id
            ), "len of postproc_fns and output_layers_id must match"

            def identity(x):
                return x

            postproc_fns = [identity if fn is None else fn for fn in postproc_fns]

        return postproc_fns

    def fit(
        self,
        model: Callable,
        fit_dataset: Optional[Union[ItemType, DatasetType]] = None,
        feature_layers_id: List[Union[int, str]] = [],
        head_layer_id: Optional[Union[int, str]] = -1,
        input_layer_id: Optional[Union[int, str]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Prepare the detector for scoring:
        * Constructs the feature extractor based on the model
        * Calibrates the detector on ID data "fit_dataset" if needed,
            using self._fit_to_dataset

        Args:
            model: model to extract the features from
            fit_dataset: dataset to fit the detector on
            feature_layers_id (List[int]): list of str or int that identify
                features to output.
                If int, the rank of the layer in the layer list
                If str, the name of the layer. Defaults to [-1]
            head_layer_id (int, str): identifier of the head layer.
                If int, the rank of the layer in the layer list
                If str, the name of the layer.
                Defaults to -1
            input_layer_id (List[int]): = list of str or int that identify the input
                layer of the feature extractor.
                If int, the rank of the layer in the layer list
                If str, the name of the layer. Defaults to None.
            verbose (bool): if True, display a progress bar. Defaults to False.
        """
        (
            self.backend,
            self.data_handler,
            self.op,
            self.FeatureExtractorClass,
        ) = import_backend_specific_stuff(model)

        # if required by the method, check that fit_dataset is not None
        if self.requires_to_fit_dataset and fit_dataset is None:
            raise ValueError(
                "`fit_dataset` argument must be provided for this OOD detector"
            )

        # react: compute threshold (activation percentiles)
        if self.use_react:
            if fit_dataset is None:
                raise ValueError(
                    "if react quantile is not None, fit_dataset must be"
                    " provided to compute react activation threshold"
                )
            else:
                self.compute_react_threshold(
                    model, fit_dataset, verbose=verbose, head_layer_id=head_layer_id
                )

        if (feature_layers_id == []) and (self.requires_internal_features):
            raise ValueError(
                "Explicitly specify feature_layers_id=[layer0, layer1,...], "
                + "where layer0, layer1,... are the names of the desired output "
                + "layers of your model. These can be int or str (even though str"
                + " is safer). To know what to put, have a look at model.summary() "
                + "with keras or model.named_modules() with pytorch"
            )

        self.feature_extractor = self._load_feature_extractor(
            model, feature_layers_id, head_layer_id, input_layer_id
        )

        if fit_dataset is not None:
            if "verbose" in inspect.signature(self._fit_to_dataset).parameters.keys():
                kwargs.update({"verbose": verbose})
            self._fit_to_dataset(fit_dataset, **kwargs)

    def _load_feature_extractor(
        self,
        model: Callable,
        feature_layers_id: List[Union[int, str]] = [],
        head_layer_id: Optional[Union[int, str]] = -1,
        input_layer_id: Optional[Union[int, str]] = None,
        return_penultimate: bool = False,
    ) -> Callable:
        """
        Loads feature extractor

        Args:
            model: a model (Keras or PyTorch) to load.
            feature_layers_id (List[int]): list of str or int that identify
                features to output.
                If int, the rank of the layer in the layer list
                If str, the name of the layer. Defaults to [-1]
            head_layer_id (int): identifier of the head layer.
                -1 when the last layer is the head
                -2 when the last layer is a softmax activation layer
                ...
                If int, the rank of the layer in the layer list
                If str, the name of the layer. Defaults to -1
            input_layer_id (List[int]): = list of str or int that identify the input
                layer of the feature extractor.
                If int, the rank of the layer in the layer list
                If str, the name of the layer. Defaults to None.
            return_penultimate (bool): if True, the penultimate values are returned,
                i.e. the input to the head_layer.

        Returns:
            FeatureExtractor: a feature extractor instance
        """
        if not self.use_ash:
            self.ash_percentile = None
        if not self.use_scale:
            self.scale_percentile = None

        feature_extractor = self.FeatureExtractorClass(
            model,
            feature_layers_id=feature_layers_id,
            input_layer_id=input_layer_id,
            head_layer_id=head_layer_id,
            react_threshold=self.react_threshold,
            scale_percentile=self.scale_percentile,
            ash_percentile=self.ash_percentile,
            return_penultimate=return_penultimate,
        )
        return feature_extractor

    def _fit_to_dataset(self, fit_dataset: DatasetType) -> None:
        """
        Fits the OOD detector to fit_dataset.

        To be overrided in child classes (if needed)

        Args:
            fit_dataset: dataset to fit the OOD detector on
        """
        raise NotImplementedError()

    def score(
        self,
        dataset: Union[ItemType, DatasetType],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs".

        Args:
            dataset (Union[ItemType, DatasetType]): dataset or tensors to score
            verbose (bool): if True, display a progress bar. Defaults to False.

        Returns:
            tuple: scores or list of scores (depending on the input) and a dictionary
                containing logits and labels.
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"
        labels = None
        # Case 1: dataset is neither a tf.data.Dataset nor a torch.DataLoader
        if isinstance(dataset, get_args(ItemType)):
            tensor = self.data_handler.get_input_from_dataset_item(dataset)
            scores = self._score_tensor(tensor)
            logits = self.op.convert_to_numpy(self.feature_extractor._last_logits)

            # Get labels if dataset is a tuple/list
            if isinstance(dataset, (list, tuple)):
                labels = self.data_handler.get_label_from_dataset_item(dataset)
                labels = self.op.convert_to_numpy(labels)

        # Case 2: dataset is a tf.data.Dataset or a torch.DataLoader
        elif isinstance(dataset, get_args(DatasetType)):
            scores = np.array([])
            logits = None

            for item in tqdm(dataset, desc="Scoring", disable=not verbose):
                tensor = self.data_handler.get_input_from_dataset_item(item)
                score_batch = self._score_tensor(tensor)
                logits_batch = self.op.convert_to_numpy(
                    self.feature_extractor._last_logits
                )

                # get the label if available
                if len(item) > 1:
                    labels_batch = self.data_handler.get_label_from_dataset_item(item)
                    labels = (
                        labels_batch
                        if labels is None
                        else np.append(labels, self.op.convert_to_numpy(labels_batch))
                    )

                scores = np.append(scores, score_batch)
                logits = (
                    logits_batch
                    if logits is None
                    else np.concatenate([logits, logits_batch], axis=0)
                )

        else:
            raise NotImplementedError(
                f"OODBaseDetector.score() not implemented for {type(dataset)}"
            )

        info = dict(labels=labels, logits=logits)
        return scores, info

    def compute_react_threshold(
        self,
        model: Callable,
        fit_dataset: DatasetType,
        verbose: bool = False,
        head_layer_id: int = -1,
    ):
        if isinstance(head_layer_id, str):
            react_layer_id = self.FeatureExtractorClass.get_layer_index_by_name(
                model, head_layer_id
            )
        else:
            react_layer_id = head_layer_id

        penult_feat_extractor = self._load_feature_extractor(
            model, head_layer_id=head_layer_id, return_penultimate=True
        )
        unclipped_features, _ = penult_feat_extractor.predict(
            fit_dataset, verbose=verbose, postproc_fns=self.postproc_fns
        )
        self.react_threshold = self.op.quantile(
            unclipped_features[0], self.react_quantile
        )

    def __call__(self, inputs: Union[ItemType, DatasetType]) -> np.ndarray:
        """
        Convenience wrapper for score

        Args:
            inputs (Union[ItemType, DatasetType]): dataset or tensors to score.
            threshold (float): threshold to use for distinguishing between OOD and ID

        Returns:
            np.ndarray: array of 0 for ID samples and 1 for OOD samples
        """
        return self.score(inputs)

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.

        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        raise NotImplementedError(
            "Property `requires_to_fit_dataset` is not implemented. It should return"
            + " a True or False boolean."
        )

    @property
    def requires_internal_features(self) -> bool:
        """
        Whether an OOD detector acts on internal model features.

        Returns:
            bool: True if the detector perform computations on an intermediate layer
            else False.
        """
        raise NotImplementedError(
            "Property `requires_internal_dataset` is not implemented. It should return"
            + " a True or False boolean."
        )
