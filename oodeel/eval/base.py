from abc import ABC, abstractmethod

class Experiment(ABC):

    def __init__(self):
        self.results = None
        self.curve = None

    @abstractmethod
    def run(self, oodmodel):
        raise NotImplementedError()

    def get_results(self):
        return self.results
