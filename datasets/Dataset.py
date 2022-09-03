from abc import ABC, abstractmethod
from math import pi
import pickle
import os
import numpy as np

class Dataset(ABC):
    @staticmethod
    def loadData(dataPath, dataEncoding ='latin1'):
        pkl_path = os.path.join(dataPath)
        f=open(pkl_path,'rb')
        return pickle.load(f,encoding = dataEncoding)
    
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