import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import convert_to_tensor

from tensorflow.python.keras.models import load_model


class OperateModel:
    @staticmethod
    def saveModel(model, path):
        model.save(path)

    @staticmethod
    def loadModel(path):
        return load_model(path)

    @staticmethod
    def _getTensorData(features, targets, testSize):
        xTrain, xTest, yTrain, yTest = train_test_split(
            features, targets, test_size=testSize, stratify=targets)
        return convert_to_tensor(xTrain), convert_to_tensor(xTest), convert_to_tensor(yTrain), convert_to_tensor(yTest)