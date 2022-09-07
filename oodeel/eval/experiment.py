from ..datasets.load import dataset_load
from ..datasets.preprocess import split #Didn't use it. Is it worth keeping ?
## preprocess might be useful to construct artificial OOD
from .base import Experiment
import oodeel 
import numpy as np
import tensorflow as tf


class SingleDSExperiment(Experiment):
    """
    Experiment where for a given dataset, k classes are considered ID
    and n-k are considered OOD (where n is the total number of classes).

    Args:
        training_func: function used for training a model (no pretrained models 
            for such a benchmark). The signature must start with "x_train, y_train".
        dataset_name: name of the dataset to split
        config: additional arguments for training_func. Defaults to None.
    """
    def __init__(self, dataset_name, id_labels=None, ood_labels=None, lim=None):
        super().__init__()
        self.dataset_name = dataset_name
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset_load(dataset_name)

        ### Construct splitting labels
        labels = np.unique(self.y_train)
        split = []
        for l in labels:
            if (id_labels is None) and (l not in ood_labels):
                split.append(l)
            elif (ood_labels is None) and (l in id_labels):
                split.append(l)
        self.split = split

        ### Construct split dataset
        id_indices_train = [1 if y in split else 0 for y in self.y_train]
        id_indices_test = [1 if y in split else 0 for y in self.y_test]
        ood_indices_test = [0 if y in split else 1 for y in self.y_test]
        self.x_train = self.x_train[np.where(id_indices_train)]
        self.y_train = self.y_train[np.where(id_indices_train)]
        self.id_dataset = self.x_test[np.where(id_indices_test)]
        self.ood_dataset = self.x_test[np.where(ood_indices_test)]
        self.y_test = self.y_test[np.where(id_indices_test)]

    def get_split_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def _train_model(self, training_fun, train_config):
        if train_config["test_data"]:
            train_config.pop("test_data")
            model = training_fun(train_data = (self.x_train, self.y_train),
                                test_data = (self.id_dataset, self.y_test),
                                **train_config)
        else:
            train_config.pop("test_data")
            model = training_fun(train_data = (self.x_train, self.y_train),
                                **train_config)
        return model




class TwoDSExperiment(Experiment):
    """
    Experiments where id_dataset is considered as ID and ood_dataset 
    is considered OOD.

    Args:
        id_dataset_name: name of the ID dataset
        ood_dataset_name: name of the OOD dataset
        oodmodel: the oodmodel to test. 
    """
    def __init__(self, id_dataset_name, ood_dataset_name, lim=None):
        super().__init__()
        self.id_dataset_name = id_dataset_name
        self.ood_dataset_name = ood_dataset_name
        _, (id_dataset, _ ) = dataset_load(id_dataset_name)
        _, (ood_dataset, _ ) = dataset_load(ood_dataset_name)
        if lim is None:
            lim = id_dataset.shape[0]
        self.id_dataset = id_dataset[:lim]
        self.ood_dataset = ood_dataset[:lim]


        
    ## todo some visus

