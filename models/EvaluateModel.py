from wsgiref.util import request_uri
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class EvaluateModel:
    def __init__(self, validationData, modelName, model = None, classes=["Baseline", "Stress", "Amusement"]):
        if (not model is None):
            self.model = model
        self.modelName = modelName
        self.featuresValidation = validationData["features"]
        self.targetValidation = validationData["targets"]
        self.classes = classes
        if (self.model._needEncoding):
            self.self.targetValidation = [ self.model._wesadThreeClassEncoder[str(label)] for label in validationData["targets"] ]
        self.predictions = None

    def _makePredictions(self):
        self.predictions = self.model._makePredictions(self.featuresValidation)

    def getClassesPredicted(self):
        if (self.predictions is None):
            return [np.argmax(probabilities) for probabilities in self._makePredictions()]
        return self.predictions

    def _getConfusionMatrix(self):
        return confusion_matrix(self.targetValidation, self.getClassesPredicted())

    def getConfusionMatrix(self):
        return pd.DataFrame(self._getConfusionMatrix(), index=self.classes, columns=self.classes)

    def printConfusionMatrix(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 30}
        plt.rc("font", **font)
        plt.figure(figsize=(30, 15))
        sns.heatmap(self.getConfusionMatrix(),
                    annot=True, cmap="Blues", fmt="d")
        plt.title('Multi-Classification Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()

    def executeEvaluation(self):
        print(f"Evaluating MoStress with model: {self.modelName}\n")
        print("Classification Report\n")
        print(classification_report(self.targetValidation,
              self.getClassesPredicted(), digits=4, target_names=self.classes))
        print("\n")
        print("Confusion Matrix\n")
        self.printConfusionMatrix()
