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

    Parameters
    ----------
    model : tf.keras model 
        keras models saved as pb files e.g. with model.save()
    """
    def __init__(
        self, 
        nearest: int = 1, 
        output_layers_id: List[int] = [-2], 
        output_activation: str = None,
        flatten: bool = True, 
        batch_size: int = 256,
        threshold: Optional[float] = None
    ):

        """
        Initializes the feature extractor 
        """
        super().__init__(output_layers_id=output_layers_id,
                         output_activation=output_activation, 
                         flatten=flatten,
                         batch_size=batch_size,
                         threshold=threshold)

        self.index = None
        self.nearest = nearest

    def _fit_to_dataset(
        self, 
        fit_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray] 
    ):
        """
        Constructs the index from ID data "fit_dataset", which will be used for
        nearest neighbor search.

        Parameters
        ----------
        fit_dataset : np.array
            input dataset (ID) to construct the index with.
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

        Parameters
        ----------
        inputs : np.array
            input samples to score

        Returns
        -------
        np.array
            scores
        """
        assert self.feature_extractor is not None, "Call .fit() before .score()"

        input_projected = self.feature_extractor(inputs)[0]
        input_projected = np.array(input_projected)
        scores, _ = self.index.search(input_projected, self.nearest)
        return scores[:,0]

        
