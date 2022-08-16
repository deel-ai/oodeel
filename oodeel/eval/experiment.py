from ..datasets.load import dataset_load
from ..datasets.preprocess import split
from .metrics import bench_metrics
## preprocess might be useful to construct artificial OOD
from .base import Experiment

import numpy as np

class SingleDSExperiment(Experiment):

    def __init__(self, dataset_name=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset = dataset_load(dataset_name)



class TwoDSExperiment(Experiment):

    def __init__(self, id_dataset_name=None, ood_dataset_name=None):
        super().__init__()
        self.id_dataset_name = id_dataset_name
        self.ood_dataset_name = ood_dataset_name
        _ , self.id_dataset = dataset_load(id_dataset_name)
        _ , self.ood_dataset = dataset_load(ood_dataset_name)

    def run(self, oodmodel, fit_dataset=None, step=4):
        if oodmodel.__class__.__mro__[1].__name__ == "OODModelWithId":
            oodmodel.fit(fit_dataset)
        id_scores = oodmodel.score(self.id_dataset[0]) ### Careful with that, have to think how to properly implement this
        ood_scores = oodmodel.score(self.ood_dataset[0])
        scores = np.concatenate([id_scores, ood_scores]) #ood has to be higher
        labels = np.concatenate([np.zeros(id_scores.shape), np.ones(ood_scores.shape)])

        self.results = bench_metrics(scores, labels, step)
        return self.results
        
    ## todo some visus

