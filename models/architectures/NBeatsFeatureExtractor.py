import datetime
import os

import matplotlib.pyplot as plt
from kerasbeats import prep_time_series
from datasets.Dataset import Dataset
from models.architectures.BaseArchitecture import BaseArchitecture
from models.architectures.NBeats.NBeatsModel import NBeatsModel
from tensorflow.python.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.python.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    ReLU,
)
from tensorflow.python.keras.models import Sequential
from tensorflow import convert_to_tensor

from utils.OperateModel import OperateModel


class NBeatsFeatureExtractor(BaseArchitecture):
    def __init__(self, moStressNeuralNetwork):
        self.moStressNeuralNetwork = moStressNeuralNetwork
        self.nBeats = NBeatsModel(lookback=7, horizon=1)
        self.residualsPath = (
            os.path.join(
                "..", "data", "preprocessedData", "residuals", "residuals.pickle"
            ),
        )
        self.residuals = (
            Dataset.loadData(self.residualsPath)
            if "residuals" in os.listdir(os.path.join("..", "data", "preprocessedData"))
            else {}
        )
        self._modelName = self.moStressNeuralNetwork._modelName
        self._modelOptimizer = self.moStressNeuralNetwork._optimizerName

    ##############---ARCHITECTURES---##############

    def setClassificationModel(self):
        model = Sequential()
        model.add(
            Dense(256, input_shape=(len(self.residuals), self.residuals[0].shape))
        )
        model.add(ReLU(alpha=0.5))
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(ReLU(alpha=0.5))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(self.moStressNeuralNetwork._numClasses))
        model.add(Activation("softmax"))
        model.summary()

        self.model = model

        return self

    ##############---OPERATIONS---##############

    def _compileModel(
        self,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    ):
        self.nBeats.compile_model()
        self.model.compile(
            optimizer=self.moStressNeuralNetwork._optimizerName,
            loss=loss,
            metrics=metrics,
        )

    def _setModelCallbacksPath(self):
        currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorBoardFilesPath = os.path.join(
            "..",
            "..",
            "logs",
            f"{self._modelName}",
            f"{self._modelOptimizer}",
            "fit",
            currentTime,
        )
        trainingCheckpointPath = os.path.join(
            "..",
            "..",
            "trainingCheckpoint",
            f"{self._modelName}",
            f"{self._modelOptimizer}",
            "cp.ckpt",
        )

        return tensorBoardFilesPath, trainingCheckpointPath

    def _setModelCallbacks(self):

        tensorBoardFilesPath, trainingCheckpointPath = self._setModelCallbacksPath()

        if not self._callbacks and not len(self._callbacks) > 0:
            self._callbacks = [
                EarlyStopping(
                    monitor="loss", patience=10, mode="min", min_delta=0.0010
                ),
                TensorBoard(
                    log_dir=tensorBoardFilesPath, write_graph=True, histogram_freq=5
                ),
                ModelCheckpoint(
                    filepath=trainingCheckpointPath, save_weights_only=True, verbose=1
                ),
            ]

        return self

    def _fitModel(self, epochs=100, shuffle=False, testSize=0.4):
        self.prepareResidualsDataset()

        xTrain, xTest, yTrain, yTest = OperateModel._getTensorData(
            self.residuals, self.moStressNeuralNetwork._allTrainTargets, testSize
        )

        self._modelName = "BASIC-CLASSIFIER"
        self._callbacks = []
        self._setModelCallbacks()

        self.moStressNeuralNetwork.history = self.model.fit(
            x=xTrain,
            y=yTrain,
            validation_data=(xTest, yTest),
            epochs=epochs,
            shuffle=shuffle,
            class_weight=self.moStressNeuralNetwork.weights,
            callbacks=self._callbacks,
        )

    def _makePredictions(self, inputData):
        return self.model.predict(x=convert_to_tensor(inputData))

    def _saveModel(self, path):
        self.model.save(path)

    def prepareResidualsDataset(self, epochs=100, verbose=False):

        if not len(self.residuals) > 0:

            for i, window in enumerate(self.moStressNeuralNetwork._allTrainFeatures):
                X, y = prep_time_series(window, lookback=7, horizon=1)
                self.nBeats.fit(
                    X, y, epochs=epochs, verbose=verbose, callbacks=self._callbacks
                )

                self.residuals[i] = self.nBeats.residualModel.predict(X)

            Dataset.saveData(self.residualsPath, self.residuals)

        return self

    def _printLearningCurves(self, loss="Sparse Categorical Crossentropy"):
        plt.figure(figsize=(30, 15))
        plt.plot(
            self.moStressNeuralNetwork.history.history["loss"],
            label=loss + " (Training Data)",
        )
        plt.plot(
            self.moStressNeuralNetwork.history.history["val_loss"],
            label=loss + " (Testing Data)",
            marker=".",
            markersize=20,
        )
        plt.title(loss)
        plt.ylabel(f"{loss} value")
        plt.xlabel("No. epoch")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()
