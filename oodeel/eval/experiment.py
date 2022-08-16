from ..datasets.load import dataset_load
from ..datasets.preprocess import split
from .metrics import bench_metrics
## preprocess might be useful to construct artificial OOD
from .base import Experiment
import oodeel 
import numpy as np
import tensorflow as tf


class SingleDSExperiment(Experiment):

    def __init__(self, training_func, dataset_name, config=None):
        super().__init__()
        self.dataset_name = dataset_name
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset_load(dataset_name)
        self.results = {}
        self.training_func = training_func
        self.config = config
        self.model = None
        self.oodmodel = None

    def run(self, oodmodel, splits, fit_dataset=None, step=4):
        """
        _summary_

        Parameters
        ----------
        oodmodel : _type_
            _description_
        splits : list or a list of list (for different spliting to be tested at once)
        fit_dataset : _type_, optional
            _description_, by default None
        step : int, optional
            _description_, by default 4
        """
        self.oodmodel = oodmodel

        if len(np.array(splits).shape) == 1:
            splits = [splits]
        for split in splits:
            id_indices_train = [1 if y in split else 0 for y in self.y_train]
            id_indices_test = [1 if y in split else 0 for y in self.y_test]
            ood_indices_test = [0 if y in split else 1 for y in self.y_test]
            x_train = self.x_train[np.where(id_indices_train)]
            y_train = self.y_train[np.where(id_indices_train)]

            if self.model is None:
                y_train = tf.keras.utils.to_categorical(y_train, len(split))
                ### Pb with to_categorical, tries to create an array of size (y.shape[0], np.max(y))
                self.model = self.training_func(x_train, y_train, self.config)

            x_id = self.x_test[np.where(id_indices_test)]
            x_ood = self.x_test[np.where(ood_indices_test)]
            self.oodmodel.model = self.model # todo change that
            if fit_dataset is not None:
                self.oodmodel.fit(fit_dataset)
            
            id_scores = self.oodmodel.score(x_id) 
            ood_scores = self.oodmodel.score(x_ood)
            scores = np.concatenate([id_scores, ood_scores]) #ood has to be higher
            labels = np.concatenate([np.zeros(id_scores.shape), np.ones(ood_scores.shape)])

            self.results[str(split)] = bench_metrics(scores, labels, step)

        return self.results





class TwoDSExperiment(Experiment):

    def __init__(self, id_dataset_name, ood_dataset_name):
        super().__init__()
        self.id_dataset_name = id_dataset_name
        self.ood_dataset_name = ood_dataset_name
        _, self.id_dataset = dataset_load(id_dataset_name)
        _, self.ood_dataset = dataset_load(ood_dataset_name)

    def run(self, oodmodel, fit_dataset=None, step=4):

        if fit_dataset is not None:
            oodmodel.fit(fit_dataset)
        id_scores = oodmodel.score(self.id_dataset[0]) ### Careful with that, have to think how to properly implement this
        ood_scores = oodmodel.score(self.ood_dataset[0])
        scores = np.concatenate([id_scores, ood_scores]) #ood has to be higher
        labels = np.concatenate([np.zeros(id_scores.shape), np.ones(ood_scores.shape)])

        self.results = bench_metrics(scores, labels, step)
        return self.results
        
    ## todo some visus

