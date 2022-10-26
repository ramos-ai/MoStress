import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
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
from utils.Logger import Logger, LogLevel

logInfo = Logger("NBeatsFeatureExtractor Architecture", LogLevel.INFO)

class NBeatsFeatureExtractor(BaseArchitecture):
    def __init__(self, moStressNeuralNetwork):
        self.moStressNeuralNetwork = moStressNeuralNetwork
        self.nBeats = NBeatsModel(lookback=7, horizon=1)
        self.residualsPath = (
            os.path.join("data", "preprocessedData", "residuals", "residuals.pickle"),
        )
        self.nBeatsSavedModelBasePath = {"models", "saved", "nBeats"}
        self.residuals = (
            Dataset.loadData(self.residualsPath)
            if "residuals" in os.listdir(os.path.join("data", "preprocessedData"))
            else {}
        )
        self._modelName = self.moStressNeuralNetwork._modelName
        self._modelOptimizer = self.moStressNeuralNetwork._optimizerName
        self._callbacks = []

    ##############---ARCHITECTURES---##############

    def classificationModel(self):
        model = Sequential()
        model.add(
            Dense(
                256, input_shape=(len(self.residuals),)
            )  # Maybe the input_shape is wrong, so maybe we would have an error on _fit because of this.
        )
        model.add(ReLU())
        model.add(Dropout(0.3))
        model.add(Dense(128))
        model.add(ReLU())
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

    def _setModelCallbacks(self):

        tensorBoardFilesPath, trainingCheckpointPath = self._setModelCallbacksPath()

        if not len(self._callbacks) > 0:
            self._callbacks = [
                EarlyStopping(
                    monitor="loss", patience=10, mode="min", min_delta=0.0010
                ),
                TensorBoard(
                    log_dir=tensorBoardFilesPath, write_graph=True, histogram_freq=5
                ),
                ModelCheckpoint(
                    filepath=trainingCheckpointPath, save_weights_only=True, verbose=0
                ),
            ]

        return self

    def _fitModel(self, epochs=100, shuffle=False, testSize=0.4):
        self._setModelCallbacks()
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
        plt.savefig("confusionMatrix.png")

    def _setModelCallbacksPath(self):
        currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorBoardFilesPath = os.path.join(
            "logs",
            f"{self._modelName}",
            f"{self._modelOptimizer}",
            "fit",
            currentTime,
        )
        trainingCheckpointPath = os.path.join(
            "trainingCheckpoint",
            f"{self._modelName}",
            f"{self._modelOptimizer}",
            "cp.ckpt",
        )

        return tensorBoardFilesPath, trainingCheckpointPath

    def prepareResidualsDataset(self, epochs=100, verbose=1):

        if not len(self.residuals) > 0:

            fullSignal = np.concatenate(self.moStressNeuralNetwork._allTrainFeatures)

            for i in range(fullSignal.shape[1]):
                logInfo(
                    f"\n NBeats Training for Time Series number {i}: Starting Now. \n"
                )
                X, y = prep_time_series(fullSignal[:, i], lookback=7, horizon=1)
                self.nBeats.fit(
                    X, y, epochs=epochs, verbose=verbose, callbacks=self._callbacks
                )
                logInfo(f"\n NBeats Training for Time Series number {i}: Finished. \n")
                logInfo(f"\n Saving Model \n")
                self.nBeats.model.save(
                    os.path.join(
                        self.nBeatsSavedModelBasePath, f"nBeatsTimeSeries_{i}.h5"
                    )
                )
                logInfo(f"\n Getting Residuals for Time Series number {i}. \n")
                for j, window in enumerate(
                    self.moStressNeuralNetwork._allTrainFeatures
                ):
                    windowResidual = {}
                    X, y = prep_time_series(window[:, i], lookback=7, horizon=1)
                    windowResidual[j] = self.nBeats.residualModel.predict(X)
                self.residuals[i] = windowResidual
                logInfo(f"\n Residuals for Time Series number {i} collected. \n")

            Dataset.saveData(self.residualsPath, self.residuals)

        return self
