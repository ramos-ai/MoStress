import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils.Logger import Logger, LogLevel
from models.architectures.NBeatsFeatureExtractor import TIME_SERIES_TO_PROCESS

logInfo = Logger("EvaluateModel", LogLevel.INFO)


class EvaluateModel:
    def __init__(
        self,
        validationData,
        modelName,
        model=None,
        classes=["Baseline", "Stress", "Amusement"],
    ):
        self.model = model
        self.modelName = modelName
        self.featuresValidation = validationData["features"]
        self.targetValidation = validationData["targets"]
        self.classes = classes
        self.predictions = None
        self.isBinaryClassification = True if len(classes) == 2 else False

    def _makePredictions(self):
        if self.predictions is None:
            self.predictions = self.model._makePredictions(self.featuresValidation)
        return self.predictions

    def getClassesPredicted(self):
        return [np.argmax(probabilities) if not self.isBinaryClassification else 0 if probabilities < 0.5 else 1 for probabilities in self._makePredictions()]

    def _getConfusionMatrix(self):
        return confusion_matrix(self.targetValidation, self.getClassesPredicted())

    def getConfusionMatrix(self):
        return pd.DataFrame(
            self._getConfusionMatrix(), index=self.classes, columns=self.classes
        )

    def printConfusionMatrix(self):
        font = {"family": "DejaVu Sans", "weight": "bold", "size": 30}
        plt.rc("font", **font)
        plt.figure(figsize=(30, 15))
        sns.heatmap(self.getConfusionMatrix(), annot=True, cmap="Blues", fmt="d")
        plt.title("Multi-Classification Confusion Matrix")
        plt.ylabel("Actual Values")
        plt.xlabel("Predicted Values")
        plt.show()
        plt.savefig(f"main/04-nbeatsFeatureExtractor/results/timeSeries{TIME_SERIES_TO_PROCESS}/confusionMatrix.png")
        

    def executeEvaluation(self):
        logInfo(f"Evaluating MoStress with model: {self.modelName}\n")
        logInfo("Classification Report\n")
        logInfo( "\n" + 
            classification_report(
                self.targetValidation,
                self.getClassesPredicted(),
                digits=4,
                target_names=self.classes,
            )
        )
        logInfo("\n")
        logInfo("Confusion Matrix\n")
        self.printConfusionMatrix()
