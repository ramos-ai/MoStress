from os.path import join

from utils.OperateModel import OperateModel
from models.ModelFactory import ModelFactory


class MoStressNeuralNetwork:
    def __init__(self, modelOpts, dataset):
        self.modelOpts = modelOpts

        self.weights = dataset["weights"]
        self._xTrain, self._xTest, self._yTrain, self._yTest = OperateModel._getTensorData(
            dataset["features"], dataset["targets"], self.modelOpts["trainingDataTestSize"])

        self._winSize = self._xTrain.shape[1]
        self._numFeatures = self._xTrain.shape[2]
        self._numClasses = len(dataset["weights"])

        self._reservoirRandomSeed = self.modelOpts["reservoirRandomSeed"]
        self._reservoirVerbosityState = self.modelOpts["reservoirVerbosityState"]
        self._allTrainFeatures = dataset["features"]
        self._allTrainTargets = dataset["targets"]

    def execute(self, modelName="REGULARIZER-LSTM", optimizer="rmsprop", modelArchitectureType="sequential"):
        '''As standard, the best model tested will be executed,
        to test other models, change methods parameters.

        The options of models available are the following: 

            OPTIONS:
                - REGULARIZER-GRU,
                - REGULARIZER-LSTM,
                - BASELINE-GRU,
                - BASELINE-LSTM,
                - BASELINE-RESERVOIR
        '''

        self._modelName = modelName
        self._optimizerName = optimizer
        self.modelFullName = f"{self._modelName}-{self._optimizerName.upper()}"

        print(f"Starting MoStress with model: {self.modelFullName}.\n")
        self.model = ModelFactory().make(self, self._modelName)

        print(f"\nCompiling model with optimizer: {self._optimizerName}\n")
        self.model._compileModel()
        print("\nModel Compiled\n")

        print("\nFitting Model\n")
        self.model._setModelCallbacks()
        self.model._fitModel()
        print("\nModel Fitted\n")

        print("\Saving Model\n")
        self.model._saveModel(
            join(
                "..",
                "models",
                "saved",
                f"{self._modelName}-{self._optimizerName.upper()}.h5"
            )
        )
        print("\nModel Saved\n")

        if (modelArchitectureType == "sequential"):
            print("\nLearning Curves\n")
            self.model._printLearningCurves()
            print("\n")
