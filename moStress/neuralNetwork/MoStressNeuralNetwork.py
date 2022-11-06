from os.path import join

from utils.OperateModel import OperateModel
from models.ModelFactory import ModelFactory
from utils.Logger import Logger, LogLevel

logInfo = Logger("MoStressNeuralNetwork Wrapper", LogLevel.INFO)

class MoStressNeuralNetwork:
    def __init__(self, modelOpts, dataset, skipAutoTrainTestSplit=False):
        self.modelOpts = modelOpts

        self.weights = dataset["weights"]
        if not skipAutoTrainTestSplit:
            (
                self._xTrain,
                self._xTest,
                self._yTrain,
                self._yTest,
            ) = OperateModel._getTensorData(
                dataset["features"],
                dataset["targets"],
                self.modelOpts["trainingDataTestSize"],
            )

        self._winSize = dataset["features"][0].shape[0]
        self._numFeatures = dataset["features"][0].shape[1]
        self._numClasses = len(dataset["weights"])

        self._reservoirRandomSeed = self.modelOpts["reservoirRandomSeed"]
        self._reservoirVerbosityState = self.modelOpts["reservoirVerbosityState"]
        self._allTrainFeatures = dataset["features"]
        self._allTrainTargets = dataset["targets"]

    def execute(
        self,
        modelName="REGULARIZER-LSTM",
        optimizer="rmsprop",
        modelArchitectureType="sequential",
    ):
        """As standard, the best model tested will be executed,
        to test other models, change methods parameters.

        The options of models available are the following:

            OPTIONS:
                - REGULARIZER-GRU,
                - REGULARIZER-LSTM,
                - BASELINE-GRU,
                - BASELINE-LSTM,
                - BASELINE-RESERVOIR
                - NBEATS-FEATURE-EXTRACTOR
        """

        self._modelName = modelName
        self._optimizerName = optimizer
        self.modelFullName = f"{self._modelName}-{self._optimizerName.upper()}"

        logInfo(f"Starting MoStress with model: {self.modelFullName}.\n")
        self.model = ModelFactory().make(self, self._modelName)

        logInfo(f"\nCompiling model with optimizer: {self._optimizerName}\n")
        self.model._compileModel()
        logInfo("\nModel Compiled\n")

        logInfo("\nFitting Model\n")
        self.model._setModelCallbacks()
        self.model._fitModel()
        logInfo("\nModel Fitted\n")

        logInfo("\Saving Model\n")
        self.model._saveModel(
            join(
                "models",
                "saved",
                f"{self._modelName}-{self._optimizerName.upper()}.h5",
            )
        )
        logInfo("\nModel Saved\n")

        if modelArchitectureType == "sequential":
            logInfo("\nLearning Curves\n")
            self.model._printLearningCurves()
            logInfo("\n")
