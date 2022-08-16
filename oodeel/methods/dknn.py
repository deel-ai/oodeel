import tensorflow as tf
from .base import OODModel, OODModelWithId
import numpy as np
from .feature_extractor import FeatureExtractor
import faiss

class DKNN(OODModelWithId):
    """
    "Out-of-Distribution Detection with Deep Nearest Neighbors"
    https://arxiv.org/abs/2204.06507

    Parameters
    ----------
    model : tf.keras model 
        keras models saved as pb files e.g. with model.save()
    """
    def __init__(self, model):
        """
        Initializes the feature extractor 
        """
        super().__init__(model)
        self.index = None
        self.feature_extractor = FeatureExtractor(model, indices=[-3])

    def fit(self, id_dataset):
        """
        Constructs the index from ID data "id_dataset", which will be used for
        nearest neighbor search.

        Parameters
        ----------
        id_dataset : np.array
            input dataset (ID) to construct the index with.
        """
        self.id_projected = self.project_id(id_dataset)
        self.index = faiss.IndexFlatL2(self.id_projected[0].shape[1])
        self.index.add(self.id_projected[0])

    def score(self, inputs, nn):
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
        inp_proj = self.project_id(inputs)
        scores, _ = self.index.search(inp_proj[0], nn)
        self.scores = -scores[:,-1]
        return self.scores

        
