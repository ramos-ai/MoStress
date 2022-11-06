import datetime
import os

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                               TensorBoard)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (GRU, LSTM, Activation, Dense,
                                            Dropout, Flatten, GaussianNoise,
                                            LeakyReLU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2

from models.architectures.BaseArchitecture import BaseArchitecture
from tensorflow import convert_to_tensor


class SequentialModels(BaseArchitecture):
    def __init__(self, moStressNeuralNetwork):
        self.moStressNeuralNetwork = moStressNeuralNetwork

    ##############---ARCHITECTURES---##############

    def gruRegularizerMoStress(self):
        return self._rnnRegularizersMoStress(GRU, L2, 0.000001, max_norm, 3)

    def lstmRegularizerMoStress(self):
        return self._rnnRegularizersMoStress(LSTM, L2, 0.000001, max_norm, 3)

    def gruBaselineMostress(self):
        return self._rnnBaselineMoStress(GRU)

    def lstmBaselineMostress(self):
        return self._rnnBaselineMoStress(LSTM)

    def _rnnRegularizersMoStress(self, neuron, regularizerFunction, regularizationFactor, biasConstraintFunction,  biasConstraintFactor):
        def regularizer(regularizerFunc): return regularizerFunc(
            regularizationFactor)

        def biasConstraint(biasFunc): return biasFunc(biasConstraintFactor)
        return self._rnnBaselineMoStress(neuron, regularizer(regularizerFunction), biasConstraint(biasConstraintFunction))

    def _rnnBaselineMoStress(self, neuron, activity_regularizer=None, bias_constraint=None):
        def rnnNeuron(neuronFunc): return neuronFunc(128, input_shape=(
            self.moStressNeuralNetwork._winSize, self.moStressNeuralNetwork._numFeatures), return_sequences=True)

        model = Sequential()
        model.add(rnnNeuron(neuron))
        model.add(LeakyReLU(alpha=0.5))
        model.add(GaussianNoise(0.5))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(self.moStressNeuralNetwork._numClasses, activity_regularizer=activity_regularizer,
                  bias_constraint=bias_constraint))
        model.add(Activation("softmax"))
        model.summary()

        self.model = model

        return self

    ##############---OPERATIONS---##############

    def _compileModel(self, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]):
        self.model.compile(
            optimizer=self.moStressNeuralNetwork._optimizerName,
            loss=loss,
            metrics=metrics   
        )

    def _setModelCallbacks(self, callbacksList=[]):
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
                EarlyStopping(monitor='sparse_categorical_accuracy',
                              patience=20, mode='min'),
                TensorBoard(log_dir=tensorBoardFilesPath,
                            write_graph=True, histogram_freq=5),
                ModelCheckpoint(filepath=trainingCheckpointPath,
                                save_weights_only=True, verbose=1)
            ]
            return
        self.moStressNeuralNetwork._callbacks = callbacksList
    
    def _fitModel(self, epochs=100, shuffle=False):
        self.moStressNeuralNetwork.history = self.model.fit(
            x=self.moStressNeuralNetwork._xTrain,
            y=self.moStressNeuralNetwork._yTrain,
            validation_data=(self.moStressNeuralNetwork._xTest,
                             self.moStressNeuralNetwork._yTest),
            epochs=epochs,
            shuffle=shuffle,
            class_weight=self.moStressNeuralNetwork.weights,
            callbacks=self.moStressNeuralNetwork._callbacks
        )
    
    def _makePredictions(self, inputData):
        return self.model.predict(x=convert_to_tensor(inputData))

    def _saveModel(self, path):
        self.model.save(path)

    def _printLearningCurves(self, loss="Sparse Categorical Crossentropy"):
        plt.figure(figsize=(30, 15))
        plt.plot(
            self.moStressNeuralNetwork.history.history['loss'], label=loss + " (Training Data)")
        plt.plot(self.moStressNeuralNetwork.history.history['val_loss'],
                 label=loss + " (Testing Data)", marker=".", markersize=20)
        plt.title(loss)
        plt.ylabel(f'{loss} value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.show()

