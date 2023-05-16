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
from tensorflow.python.keras.regularizers import L2
from tensorflow.python.keras.constraints import max_norm
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor
from sklearn.utils.class_weight import compute_class_weight
from models.architectures.NBeats.blocks.NBeatsBlock import NBeatsBlock

from utils.OperateModel import OperateModel
from utils.Logger import Logger, LogLevel
from utils.utils import createFolder, hasDataFile, deleteData
import pandas as pd
import numpy as np
import gc

logInfo = Logger("NBeatsFeatureExtractor Architecture", LogLevel.INFO)
logWarning = Logger("NBeatsFeatureExtractor Architecture", LogLevel.WARN)
logDebug = Logger("NBeatsFeatureExtractor Architecture", LogLevel.DEBUG)

ROUNDS = 5
BATCH_TO_PROCESS = 4
TIME_SERIES_TO_PROCESS = 0

class NBeatsFeatureExtractor(BaseArchitecture):

    def __init__(self, moStressNeuralNetwork):
        self.moStressNeuralNetwork = moStressNeuralNetwork
        self.nBeats = NBeatsModel(lookback=7, horizon=1)
        self.residualsFolderPath = os.path.join("data", "preprocessedData",
                                                "residuals")
        self.residualsTrainingFolderPath = os.path.join("data", "preprocessedData",
                                                        "residuals", "training")
        self.residualsValidationFolderPath = os.path.join("data", "preprocessedData",
                                                          "residuals", "validation")
        self.nBeatsSavedModelBasePath = os.path.join("models", "saved",
                                                     "nBeats")
        self.basicClassifierModelPath = os.path.join(
            self.nBeatsSavedModelBasePath,
            f"timeSeries{TIME_SERIES_TO_PROCESS}",
            f"basicClassifier_{TIME_SERIES_TO_PROCESS}.h5"
        )
        self.resultsPath = os.path.join(
            "main",
            "04-nbeatsFeatureExtractor",
            "results"
        )
        self.currentTimeSeriesResult = os.path.join(
            self.resultsPath,
            f"timeSeries{TIME_SERIES_TO_PROCESS}"
        )
        self.trainingDataPath = os.path.join(
            self.residualsFolderPath,
            f"timeSeries{TIME_SERIES_TO_PROCESS}",
            f"batch{BATCH_TO_PROCESS}",
        )
        createFolder(self.residualsFolderPath)
        createFolder(self.residualsTrainingFolderPath)
        createFolder(self.residualsValidationFolderPath)
        createFolder(self.nBeatsSavedModelBasePath)
        createFolder(self.resultsPath)
        createFolder(self.currentTimeSeriesResult)

        # self.residuals = Dataset.loadData(
        #     os.path.join(
        #         self.residualsTrainingFolderPath,
        #         f"residualTimeSeries_{TIME_SERIES_TO_PROCESS}_{BATCH_TO_PROCESS}.pickle"
        #     )
        # )

        self._modelName = self.moStressNeuralNetwork._modelName
        self._modelOptimizer = self.moStressNeuralNetwork._optimizerName
        self._callbacks = []

        self.hasToCollectResiduals = False
        self.hasToFitBasicClassifier = True

    ############## ---ARCHITECTURES---##############

    def classificationModel(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(30, 419)))
        model.add(ReLU())
        model.add(Dropout(0.7))
        # model.add(Dense(32))
        # model.add(ReLU())
        # model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(self.moStressNeuralNetwork._numClasses, activity_regularizer=L2(0.000001), bias_constraint=max_norm(3)))
        model.add(Activation("softmax"))
        model.summary()

        self.model = model

        return self

    ############## ---OPERATIONS---##############

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
                EarlyStopping(monitor="val_loss",
                              patience=20,
                              mode="min"),
                TensorBoard(log_dir=tensorBoardFilesPath,
                            write_graph=True,
                            histogram_freq=5),
                ModelCheckpoint(filepath=trainingCheckpointPath,
                                save_weights_only=True,
                                verbose=0),
                logInfo,
            ]

        return self

    def _fitModel(self, epochs=100, shuffle=False, testSize=0.4):
        if (self.hasToCollectResiduals):
            self._setModelCallbacks()
            self.prepareResidualsDataset()

        
        # qtdWindows = len(self.moStressNeuralNetwork._allTrainTargets)
        # maxIndex = qtdWindows - 1
        # indexSlicer = maxIndex//3
        # targets = self.moStressNeuralNetwork._allTrainTargets[(BATCH_TO_PROCESS-1)*indexSlicer : (BATCH_TO_PROCESS)*indexSlicer]

        for roundIndex in range(ROUNDS):
            logInfo(f"Starting Round {roundIndex + 1}")
            for batchIndex in range(BATCH_TO_PROCESS):
                logInfo(f"Starting Batch {batchIndex + 1}")
                trainingDataPath = os.path.join(
                self.residualsFolderPath,
                f"timeSeries{TIME_SERIES_TO_PROCESS}",
                f"batch{batchIndex+1}",
                )

                if self.hasToFitBasicClassifier:
                    if hasDataFile(self.basicClassifierModelPath):
                        self.model = load_model(self.basicClassifierModelPath)

                features = Dataset.loadData(os.path.join(trainingDataPath, "features.pickle"))
                targets = Dataset.loadData(os.path.join(trainingDataPath, "targets.pickle"))

                xTrain, xTest, yTrain, yTest = OperateModel._getTensorData(features, targets, testSize)
                deleteData(features)
                deleteData(targets)

                self._modelName = f"BASIC-CLASSIFIER-TIME-SERIES-{TIME_SERIES_TO_PROCESS}-{batchIndex}"
                self._callbacks = []
                self._setModelCallbacks()

                weights = compute_class_weight(class_weight="balanced", classes=np.unique(targets), y=targets)
                weights = {i: weights[i] for i in range(len(weights))}

                self.moStressNeuralNetwork.history = self.model.fit(
                    x=xTrain,
                    y=yTrain,
                    validation_data=(xTest, yTest),
                    epochs=epochs,
                    shuffle=shuffle,
                    class_weight=weights,
                    callbacks=self._callbacks,
                )
                deleteData(xTrain)
                deleteData(yTrain)
                deleteData(xTest)
                deleteData(yTest)

                pathToSaveHistory = os.path.join(
                    self.nBeatsSavedModelBasePath,
                    f"timeSeries{TIME_SERIES_TO_PROCESS}",
                    f"history_{batchIndex+1}_{roundIndex+1}.pickle"
                )

                Dataset.saveData(
                    pathToSaveHistory,
                    self.moStressNeuralNetwork.history.history
                )

                self.model.save(self.basicClassifierModelPath)

    def _makePredictions(self, inputData):
        return self.model.predict(x=convert_to_tensor(inputData))

    def _saveModel(self, path):
        if hasDataFile(self.basicClassifierModelPath):
            return
        path = self.basicClassifierModelPath
        self.model.save(path)

    def _printLearningCurves(self, loss="Sparse Categorical Crossentropy"):
        if self.hasToFitBasicClassifier:
            self._getLossLearningCurve(loss)
            self._getMetricLearningCurve()
        

    def _getLossLearningCurve(self, loss):
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
        plt.savefig(os.path.join(self.currentTimeSeriesResult, f"lossLearningCurve_{TIME_SERIES_TO_PROCESS}_{BATCH_TO_PROCESS}"))
    
    def _getMetricLearningCurve(self):
        plt.figure(figsize=(30, 15))
        plt.plot(
            self.moStressNeuralNetwork.history.history["sparse_categorical_accuracy"],
            label="Sparse Categorical Accuracy" + " (Training Data)",
        )
        plt.plot(
            self.moStressNeuralNetwork.history.history["val_sparse_categorical_accuracy"],
            label= "Sparse Categorical Accuracy" + " (Testing Data)",
            marker=".",
            markersize=20,
        )
        plt.title("Sparse Categorical Accuracy")
        plt.ylabel(f"Sparse Categorical Accuracy value")
        plt.xlabel("No. epoch")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.currentTimeSeriesResult, f"metricLearningCurve_{TIME_SERIES_TO_PROCESS}_{BATCH_TO_PROCESS}"))
    


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

            logInfo(f"Starting the training of Nbeats model.")
            self._trainNbeatsModel(epochs, verbose)
            logInfo(f"All Nbeats models trained and saved.")

            logInfo(f"Starting to generate the residuals dataset")
            self._generateResiduals(epochs, verbose)
            logInfo(f"All residuals collected")

    def _trainNbeatsModel(self, epochs, verbose):

        fullSignal = np.concatenate(
            self.moStressNeuralNetwork._allTrainFeatures)

        for i in range(fullSignal.shape[1]):

            timeSeriesPath = os.path.join(self.nBeatsSavedModelBasePath,
                                          f"timeSeries{i}")
            createFolder(timeSeriesPath)

            if len(os.listdir(timeSeriesPath)) < 2:

                logInfo(
                    f"\n NBeats Training for Time Series number {i}: Starting Now. \n"
                )
                X, y = prep_time_series(fullSignal[:, i],
                                        lookback=7,
                                        horizon=1)
                self.nBeats.fit(X,
                                y,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=self._callbacks)
                logInfo(
                    f"\n NBeats Training for Time Series number {i}: Finished. \n"
                )

                logInfo(f"\n Saving Model \n")
                self.nBeats.model.save(
                    os.path.join(timeSeriesPath, f"nBeatsTrainedModel_{i}.h5"))
                self.nBeats.residualModel.save(
                    os.path.join(timeSeriesPath, f"residualModel_{i}.h5"))
                logInfo(f"\n Model Saved \n")

    def _generateResiduals(self):

        for i in range(self.moStressNeuralNetwork._numFeatures):

            timeSeriesModelPath = os.path.join(self.nBeatsSavedModelBasePath,
                                               f"timeSeries{i}",
                                               f"residualModel_{i}.h5")
            savedResidualPath = os.path.join(self.residualsFolderPath,
                                             f"residualTimeSeries_{i}.pickle")

            logInfo(f"\n Getting Residuals for Time Series number {i}. \n")
            try:
                logDebug(
                    f"\n Checking if the residuals of Time Series number {i} exists. \n")
                try:
                    windowResidual = Dataset.loadData(savedResidualPath)
                    logDebug(
                        f"\n Residuals of of Time Series number {i} already exists. Moving to the next residual. \n")
                except:
                    logDebug(
                        f"\n Residuals of of Time Series number {i} don't exists. Starting collection. \n")
                    residualModel = load_model(timeSeriesModelPath,
                                               {'NBeatsBlock': NBeatsBlock})

                    for window in self.moStressNeuralNetwork._allTrainFeatures:
                        windowResidual = []
                        X, _ = prep_time_series(window[:, i],
                                                lookback=7,
                                                horizon=1)
                        windowResidual.append(residualModel.predict(X))

                self.residuals[i] = windowResidual
                Dataset.saveData(savedResidualPath, windowResidual)

                logInfo(
                    f"\nResiduals for Time Series number {i}, collected with success. \n"
                )

            except Exception as e:
                logWarning(
                    f"It was not possible to collect the residuals for Time Series {i}. Error {e}"
                )

    @staticmethod
    def collectValidationResiduals(residualFeatures):

        folderToSaveDataPath = os.path.join(
            "data",
            "preprocessedData",
            "residuals",
            "validation"
        )

        nBeatsFolderPath = os.path.join(
            "models",
            "saved",
            "nBeats",
        )

        if len(os.listdir(folderToSaveDataPath)) >= 5:
            logInfo("Residuals has been already collected.")
            return

        def adjustTimeSeries(npArray):
            timeSeries = np.concatenate(
                [npArray[0, :-1], npArray[:, -1]], 0
            )
            return convert_to_tensor(timeSeries)

        def getListOfTensor(windowList):
            return pd.DataFrame(windowList).applymap(adjustTimeSeries).to_numpy().tolist()

        for timeSeriesIndex in range(residualFeatures[0].shape[1]):
            logInfo(
                f"Starting collection of validation residual of time series {timeSeriesIndex}")
            windowResidual = []
            modelPath = os.path.join(
                nBeatsFolderPath,
                f"timeSeries{timeSeriesIndex}",
                f"residualModel_{timeSeriesIndex}.h5"
            )
            residualPath = os.path.join(
                folderToSaveDataPath,
                f"residualTimeSeries_{timeSeriesIndex}.pickle"
            )
            if (hasDataFile(residualPath)):
                logInfo(
                    f"Residual of time series {timeSeriesIndex}, already collected, moving to the next one.")
                continue
            model = load_model(modelPath, {'NBeatsBlock': NBeatsBlock})
            logDebug(f"Model loaded")

            for window in residualFeatures:
                X, _ = prep_time_series(
                    window[:, timeSeriesIndex], lookback=7, horizon=1)
                windowResidual.append(model.predict(X))
                del (X)
                gc.collect()
            residualData = getListOfTensor(windowResidual)
            del (windowResidual)
            gc.collect()
            logInfo(f"Saving residual.")
            Dataset.saveData(
                residualPath,
                residualData
            )
            logInfo(f"Residual Saved.")
            del (residualData)
            gc.collect()
