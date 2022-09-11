from reservoirpy import set_seed, verbosity
from reservoirpy.nodes import Reservoir, Ridge, Input
import numpy as np

from models.architectures.BaseArchitecture import BaseArchitecture

class ReservoirModels(BaseArchitecture):
    def __init__(self, moStressNeuralNetwork):
        self.moStressNeuralNetwork = moStressNeuralNetwork
        set_seed(moStressNeuralNetwork._reservoirRandomSeed)
        verbosity(moStressNeuralNetwork._reservoirVerbosityState)
        self._xTrain = moStressNeuralNetwork.dataset["features"]
        self._yTrain = moStressNeuralNetwork.dataset["targets"]

    ##############---ARCHITECTURES---##############

    def baseline(self, numNodes = 128, spectralRatio = 0.9, leakingRate = 0.1, ridgeFactor = 1e-6):
        self._source = Input()
        self._reservoir = Reservoir(numNodes, sr=spectralRatio, lr=leakingRate)
        self._readout = Ridge(ridge=ridgeFactor)

        self._model = self._source >> self._reservoir >> self._readout

        return self

    ##############---OPERATIONS---##############

    def _compileModel(self):
        pass
    
    def _setModelCallbacks(self):
        pass

    def _fitModel(self):
        self._statesTrain = []
        print("\nSetting Reservoir Nodes with Input Data...\n")
        for x in self._xTrain:
            states = self._reservoir.run(x, reset=True)
            self._statesTrain.append(states[-1, np.newaxis])
        print("\nFitting output nodes\n")
        self._readout.fit(self._statesTrain, self._yTrain)
    
    def _makePredictions(self, inputData):
        self._yPred = []
        print("\nSetting Reservoir Nodes with Input Data...\n")
        for x in inputData:
            self._predStates = self._reservoir.run(x, reset=True)
            y = self._readout.run(self._predStates[-1, np.newaxis])
            self._yPred.append(y)
        print("\nReturning Predictions\n")
        
        return [ np.argmax(y) for y in self._yPred ]
