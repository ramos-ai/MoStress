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

    def execute(self, modelName="REGULARIZER-LSTM", optimizer="rmsprop"):
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
        OperateModel.saveModel(
            self.model,
            join(
                "..",
                "models",
                "saved",
                f"{self._modelName}-{self._optimizerName.upper()}.h5"
            )
        )
        print("\nModel Saved\n")

        print("\nLearning Curves\n")
        self.model._printLearningCurves()
        print("\n")
