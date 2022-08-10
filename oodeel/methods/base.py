from typing import Type, Union, Iterable, Callable
import tensorflow as tf
from abc import ABC, abstractmethod

class OODModel(ABC):

    def __init__(self, model):

        self.threshold = None
        self.model = model
        self.feature_extractor = None
        self.scores = None


    @abstractmethod
    def score(self, inputs):

        raise NotImplementedError()

    def isood(self, inputs=None):

        if (self.scores is None) or (inputs is not None):
            self.score(inputs)
        oodness = tf.map_fn(lambda x: 0 if x < self.threshold else 1, self.scores)

        return oodness

    def __call__(self, inputs):

        return self.isood(inputs)
            

class OODModelWithId(OODModel):
    """
    Model that uses ID data to score new input data
    """
    def __init__(self, model, id_dataset=None):
        super().__init__(model)
        self.id_projected = None

    def project_id(self, inputs):
        id_projected = self.feature_extractor(inputs)
        return id_projected

    @abstractmethod
    def fit(self, id_dataset):

        raise NotImplementedError()




    


