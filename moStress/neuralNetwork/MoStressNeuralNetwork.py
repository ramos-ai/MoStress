from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor
from models.architectures.AchitectureFactory import ArchitectureFactory
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
import datetime

class MoStressNeuralNetwork:
    def __init__(self, modelOpts, dataset):
        self.modelOpts = modelOpts

        self.weights = dataset["weights"]
        self._getTensorData(dataset["features"], dataset["targets"])

        self._winSize = self._xTrain.shape[1]
        self._numFeatures = self._xTrain.shape[2]
        self._numClasses = len(dataset["weights"])


    def _getTensorData(self, features, targets):
        xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size=self.modelOpts["trainningDataTestSize"], stratify=targets)
        self._xTrain = convert_to_tensor(xTrain)
        self._xTest = convert_to_tensor(xTest)
        self._yTrain = convert_to_tensor(yTrain, dtype="int64")
        self._yTest = convert_to_tensor(yTest, dtype="int64")

    def _createModel(self, modelName):
        '''
            OPTIONS:
                - REGULARIZER-GRU,
                - REGULARIZER-LSTM,
                - BASELINE-GRU,
                - BASELINE-LSTM,
        '''
        modelOpts = {
            "winSize": self._winSize,
            "numFeatures": self._numFeatures,
            "numClasses": self._numClasses,
        }
        self._modelName = modelName
        self.model = ArchitectureFactory().make(self._modelName, modelOpts)

    def _compileModel(self, optimizer, loss="sparse_categorical_crossentropy" , metrics=["sparse_categorical_accuracy"]):
        self._optimizerName = optimizer
        self.model.compile(
            optimizer=self._optimizerName,
            loss=loss,
            metrics=metrics
        )
    
    def _fitModel(self, epochs = 100, shuffle = False):
        self.history = self.model.fit(
        x=self._xTrain,
        y=self._yTrain,
        validation_data=(self._xTest, self._yTest),
        epochs=epochs,
        shuffle=shuffle,
        class_weight=self.weights,
        callbacks=self._callbacks
    )
    
    def _setModelCallbacks(self, callbacksList = []):
        if (not len(callbacksList) > 0):
            self._callbacks = [
                EarlyStopping(monitor='sparse_categorical_accuracy', patience=20, mode='min'),
                TensorBoard(log_dir = f"logs/{self._modelName}/{self._optimizerName}/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), write_graph=True, histogram_freq=5)
            ]
            return
        self._callbacks = callbacksList
    
    def execute(self, modelName = "REGULARIZER-LSTM", optimizer = "rmsprop"):
        '''As standard, the best model tested will be executed,
        to test other models, change methods parameters.'''

        self._modelName = modelName
        self._optimizerName = optimizer

        print(f"Starting MoStress with model architecture: {self._modelName} and optimizer: {self._optimizerName}.\n")
        print(f"\nCreating model: {self._modelName}\n")
        self._createModel(self._modelName)
        print("\nModel Created\n")
        print(f"\nCompiling model with optmizer: {self._optimizerName}\n")
        self._compileModel(self._optimizerName)
        print("\nModel Compiled\n")
        print("\nFitting Model\n")
        self._setModelCallbacks()
        self._fitModel()
        print("\nModel Fitted\n")



