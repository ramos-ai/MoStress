import pickle
from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    @staticmethod
    def loadData(dataPath, dataEncoding='latin1'):
        f = open(dataPath, 'rb')
        return pickle.load(f, encoding=dataEncoding)

    @staticmethod
    def saveData(dataPath, data):
        with open(dataPath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _adjustUnnecessaryLabelCode(self, labelData, oldIndex, newIndex):
        oldIndexPosition = np.where(labelData == oldIndex)
        labelData[oldIndexPosition] = newIndex

    @abstractmethod
    def _getData(self):
        pass
