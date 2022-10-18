import tensorflow as tf
from .base import OODModel
import numpy as np
import faiss
from ..types import *

class DKNN(OODModel):
    """
    "Out-of-Distribution Detection with Deep Nearest Neighbors"
    https://arxiv.org/abs/2204.06507
    Simplified version adapted to convnet as built in ./models/train/train_mnist.py

        Args:
            nearest: number of nearest neighbors to consider. 
                Defaults to 1.
            output_layers_id: feature space on which to compute nearest neighbors. 
                Defaults to [-2].
            output_activation: output activation to use. 
                Defaults to None.
            flatten: Flatten the output features or not. 
                Defaults to True.
            batch_size: batch_size used to compute the features space
                projection of input data. 
                Defaults to 256.
    """
    def __init__(
        self, 
        nearest: int = 1, 
        output_layers_id: List[int] = [-2], 
        output_activation: str = None,
        flatten: bool = True, 
        batch_size: int = 256,
    ):
        super().__init__(output_layers_id=output_layers_id,
                         output_activation=output_activation, 
                         flatten=flatten,
                         batch_size=batch_size)

        self.index = None
        self.nearest = nearest

    def _fit_to_dataset(
        self, 
        fit_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray] 
    ):
        """
        Constructs the index from ID data "fit_dataset", which will be used for
        nearest neighbor search.

        Args:
            fit_dataset: input dataset (ID) to construct the index with.
        """
        fit_projected = self.feature_extractor(fit_dataset)[0]
        self.index = faiss.IndexFlatL2(fit_projected.shape[1])
        self.index.add(fit_projected)

    def _score_tensor(
        self, 
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Computes an OOD score for input samples "inputs" based on 
        the distance to nearest neighbors in the feature space of self.model

        Args:
            inputs: input samples to score

        Returns:
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"

        input_projected = self.feature_extractor(inputs)[0]
        input_projected = np.array(input_projected)
        scores, _ = self.index.search(input_projected, self.nearest)
        return scores[:,0]

        
