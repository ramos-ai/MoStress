from random import randint, seed

import numpy as np
from moStress.preprocessing.implementedSteps.Steps import Steps
from sklearn.utils.class_weight import compute_class_weight


class WeightsCalculation(Steps):
    def __init__(self, moStressPreprocessing):
        self.moStressPreprocessing = moStressPreprocessing
        self._hasWeightsCalculationFinished = False

    def execute(self):
        try:
            if (not self._hasWeightsCalculationFinished):
                self.moStressPreprocessing.featuresModelValidation, self.moStressPreprocessing.targetsModelValidation = self._getValidationData()
                self.moStressPreprocessing.features, self.moStressPreprocessing.targets = self._getTrainingData()
                self.moStressPreprocessing.weights = self._getWeights()
                self._hasWeightsCalculationFinished = True
        except:
            raise Exception("Weights Calculation failed.")

    def _getValidationData(self):
        seed(124816)
        indexToRemove = randint(
            0, self.moStressPreprocessing.quantityOfSets - 1)
        return self.moStressPreprocessing.features.pop(indexToRemove), self.moStressPreprocessing.targets.pop(indexToRemove)

    def _getTrainingData(self):
        trainingFeatures = self.moStressPreprocessing.features[0]
        trainingTargets = self.moStressPreprocessing.targets[0]
        for i in range(1, self.moStressPreprocessing.quantityOfSets - 1):
            trainingFeatures += self.moStressPreprocessing.features[i]
            trainingTargets += self.moStressPreprocessing.targets[i]
        return trainingFeatures, trainingTargets

    def _getWeights(self):
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.moStressPreprocessing.targets),
            y=self.moStressPreprocessing.targets
        )
        weights = {i: weights[i] for i in range(len(weights))}
        return weights
