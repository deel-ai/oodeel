from abc import ABC, abstractmethod
from .metrics import bench_metrics, get_curve
import numpy as np
from functools import partial

class Experiment(ABC):
    """
    Abstract class for OOD evaluation experiments
    """

    def __init__(self):
        self.curves = None

    def _train_model(self, training_fun, train_config):
        raise NotImplementedError()

    def run(self, oodmodel, model=None, fit_dataset=None, step=4, 
            training_fun=None, train_config=None):
        """
        Runs the benchmark

        Args:
            model: model to initialize oodmodel with
            oodmodel: OOD method to test
            fit_dataset: if the OOD method needs to be fit to ID data. Defaults to None.
            step: integration step (wrt percentile).. Defaults to 4.

        Returns:
            the output of the function bench_metrics. 
            (matric1, metric2,...), (true positive curve, false positive curve,...)
        """
    
        if model is not None:
            oodmodel.load(model, fit_dataset)
        else:
            model = self._train_model(training_fun, train_config)
            oodmodel.load(model, fit_dataset)

        id_scores = oodmodel.score(self.id_dataset) 
        ood_scores = oodmodel.score(self.ood_dataset)
        scores = np.concatenate([id_scores, ood_scores]) #ood has to be higher
        labels = np.concatenate([np.zeros(id_scores.shape), np.ones(ood_scores.shape)])

        self.curves = get_curve(scores, labels, step=4)
        results = bench_metrics(self.curves, metrics=["auroc"])
        return results

    def get_metric(self, metric):
        assert self.curves is not None, "Call .run() first"
        results = bench_metrics(self.curves[1], metrics=[metric]) 
        return results



