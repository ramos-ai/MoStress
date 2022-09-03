from os.path import join

from moStress.neuralNetwork.modelHandler.OperateModel import OperateModel


class MoStressNeuralNetwork:
    def __init__(self, modelOpts, dataset):
        self.modelOpts = modelOpts

        self.weights = dataset["weights"]
        self._xTrain, self._xTest, self._yTrain, self._yTest = OperateModel._getTensorData(
            dataset["features"], dataset["targets"], self.modelOpts["trainningDataTestSize"])

        self._winSize = self._xTrain.shape[1]
        self._numFeatures = self._xTrain.shape[2]
        self._numClasses = len(dataset["weights"])

    def execute(self, modelName="REGULARIZER-LSTM", optimizer="rmsprop"):
        '''As standard, the best model tested will be executed,
        to test other models, change methods parameters.'''

        self._modelName = modelName
        self._optimizerName = optimizer
        self.modelFullName = f"{self._modelName}-{self._optimizerName.upper()}"

        modelOperator = OperateModel(self)

        print(
            f"Starting MoStress with model architecture: {self._modelName} and optimizer: {self._optimizerName}.\n")
        print(f"\nCreating model: {self._modelName}\n")
        modelOperator._createModel(self._modelName)
        print("\nModel Created\n")
        print(f"\nCompiling model with optimizer: {self._optimizerName}\n")
        modelOperator._compileModel(self._optimizerName)
        print("\nModel Compiled\n")
        print("\nFitting Model\n")
        modelOperator._setModelCallbacks()
        modelOperator._fitModel()
        print("\nModel Fitted\n")
        print("\Saving Model\n")
        OperateModel.saveModel(self.model, join(
            "models", "saved", f"{self._modelName}-{self._optimizerName.upper()}.h5"))
        print("\nModel Saved\n")
        print("\nLearning Curves\n")
        modelOperator._printLearningCurves()
        print("\n")
