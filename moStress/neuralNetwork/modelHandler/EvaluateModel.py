from tensorflow import convert_to_tensor
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluateModel:
    def __init__(self, model, modelName, validationData, classes = [ "Baseline", "Stress", "Amusement" ]):
        self.model = model
        self.modelName = modelName
        self.featuresValidation = validationData["features"]
        self.targetValidation = validationData["targets"]
        self.classes = classes

    def _makePredictions(self):
        return self.model.predict( x = convert_to_tensor(self.featuresValidation) )
    
    def getClassesPredicted(self):
        return [ np.argmax(probabilities) for probabilities in self._makePredictions() ]
    
    def _getConfusionMatrix(self):
        return confusion_matrix(self.targetValidation, self.getClassesPredicted())
    
    def getConfusionMatrix(self):
        return pd.DataFrame(self._getConfusionMatrix(), index = self.classes, columns = self.classes)
    
    def printConfusionMatrix(self):
        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 30}
        plt.rc("font",**font)
        plt.figure(figsize=(30,15))
        sns.heatmap(self.getConfusionMatrix(), annot=True, cmap="Blues", fmt="d")
        plt.title('Multi-Classification Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()
    
    def executeEvaluation(self):
        print(f"Evaluating MoStress with model: {self.modelName}\n")
        print("Classification Report\n")
        print(classification_report(self.targetValidation, self.getClassesPredicted()))
        print("\n")
        print("Confusion Matrix\n")
        self.printConfusionMatrix()