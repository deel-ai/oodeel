##from ..datasets import load, preprocess
## preprocess might be useful to construct artificial OOD
from .base import Experiment

"""
class SingleDSExperiment(Experiment):

    def __init__(self, dataset_name=None):
        super.__init__(self)
        self.dataset_name = dataset_name
        self.dataset = load.dataset_load(id_dataset)



class TwoDSExperiment(Experiment):

    def __init__(self, id_dataset_name=None, ood_dataset_name=None):
        super.__init__(self)
        self.id_dataset_name = id_dataset_name
        self.ood_dataset_name = ood_dataset_name
        self.id_dataset = load.dataset_load(id_dataset)
        self.ood_dataset = load.dataset_load(ood_dataset)
"""