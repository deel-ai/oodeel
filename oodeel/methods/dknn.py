import tensorflow as tf
from .base import OODModel, OODModelWithId
import numpy as np
from ..utils.feature_extractor import FeatureExtractor
import faiss

class DKNN(OODModelWithId):

    def __init__(self, model):
        super().__init__(model)

        self.index = None
        self.feature_extractor = FeatureExtractor(model, indices=[-1])

    def fit(self, id_dataset):
        self.id_projected = self.project_id(id_dataset)
        self.index = faiss.IndexFlatL2(self.id_projected[0].shape[1])
        self.index.add(self.id_projected[0])

    def score(self, inputs, nn):
        inp_proj = self.project_id(inputs)
        scores, _ = self.index.search(inp_proj[0], nn)
        self.scores = -scores[:,-1]
        return self.scores

        
