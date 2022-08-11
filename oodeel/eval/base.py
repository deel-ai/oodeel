from abc import ABC, abstractmethod

class Experiment(ABC):

    def __init__(self):
        self.results = None

    @abstractmethod
    def run(self, model, method):
        raise NotImplementedError()

    def get_results(self):
        return self.results
