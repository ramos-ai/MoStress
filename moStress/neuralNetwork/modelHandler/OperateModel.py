from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import datetime
import os

from models.architectures.ArchitectureFactory import ArchitectureFactory

class OperateModel:
    def __init__(self, moStressNeuralNetwork):
        self.moStressNeuralNetwork = moStressNeuralNetwork
    
    @staticmethod
    def saveModel(model, path):
        model.save(path)

    @staticmethod
    def loadModel(path):
        return load_model(path)
    
    @staticmethod
    def _getTensorData(features, targets,testSize):
        xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size=testSize, stratify=targets)
        return convert_to_tensor(xTrain), convert_to_tensor(xTest), convert_to_tensor(yTrain), convert_to_tensor(yTest)
    
    def _createModel(self, modelName):
        '''
            OPTIONS:
                - REGULARIZER-GRU,
                - REGULARIZER-LSTM,
                - BASELINE-GRU,
                - BASELINE-LSTM,
        '''
        self.moStressNeuralNetwork._modelName = modelName
        self.moStressNeuralNetwork.model = ArchitectureFactory().make(
            self.moStressNeuralNetwork._modelName,
            {
                "winSize": self.moStressNeuralNetwork._winSize,
                "numFeatures": self.moStressNeuralNetwork._numFeatures,
                "numClasses": self.moStressNeuralNetwork._numClasses,
            }
        )
    
    def _compileModel(self, optimizer, loss="sparse_categorical_crossentropy" , metrics=["sparse_categorical_accuracy"]):
        self.moStressNeuralNetwork._loss = loss
        self.moStressNeuralNetwork._optimizerName = optimizer
        self.moStressNeuralNetwork.model.compile(
            optimizer=self.moStressNeuralNetwork._optimizerName,
            loss=loss,
            metrics=metrics
        )
    
    def _fitModel(self, epochs = 100, shuffle = False):
        self.moStressNeuralNetwork.history = self.moStressNeuralNetwork.model.fit(
        x=self.moStressNeuralNetwork._xTrain,
        y=self.moStressNeuralNetwork._yTrain,
        validation_data=(self.moStressNeuralNetwork._xTest, self.moStressNeuralNetwork._yTest),
        epochs=epochs,
        shuffle=shuffle,
        class_weight=self.moStressNeuralNetwork.weights,
        callbacks=self.moStressNeuralNetwork._callbacks
    )

    def _setModelCallbacks(self, callbacksList = []):
        currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorBoardFilesPath = os.path.join(
            "logs",
            f"{self.moStressNeuralNetwork._modelName}",
            f"{self.moStressNeuralNetwork._optimizerName}",
            "fit",
            currentTime
        )
        
        trainingCheckpointPath = os.path.join(
            "trainingCheckpoint",
            f"{self.moStressNeuralNetwork._modelName}",
            f"{self.moStressNeuralNetwork._optimizerName}",
            "cp.ckpt"
        )

        if (not len(callbacksList) > 0):
            self.moStressNeuralNetwork._callbacks = [
                EarlyStopping(monitor='sparse_categorical_accuracy', patience=20, mode='min'),
                TensorBoard(log_dir=tensorBoardFilesPath, write_graph=True, histogram_freq=5),
                ModelCheckpoint(filepath=trainingCheckpointPath, save_weights_only=True, verbose=1)
            ]
            return
        self.moStressNeuralNetwork._callbacks = callbacksList

    def _printLearningCurves(self, loss = "Sparse Categorical Crossentropy"):
        plt.figure(figsize=(30,15))
        plt.plot(self.moStressNeuralNetwork.history.history['loss'], label = loss + " (Trainning Data)")
        plt.plot(self.moStressNeuralNetwork.history.history['val_loss'], label = loss + " (Testing Data)", marker=".", markersize=20)
        plt.title(loss)
        plt.ylabel(f'{loss} value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()