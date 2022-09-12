from abc import ABC, abstractmethod

class BaseArchitecture(ABC):
    @abstractmethod
    def _compileModel(self):
        pass

    @abstractmethod
    def _setModelCallbacks(self):
        pass

    @abstractmethod
    def _fitModel(self):
        pass

    @abstractmethod
    def _makePredictions(self):
        pass

    @abstractmethod
    def _saveModel(self):
        pass