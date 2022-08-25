from moStress.preprocessing.implementedSteps.Steps import Steps
import numpy as np
from random import seed, randint
from sklearn.utils.class_weight import compute_class_weight

class WeightsCalculation(Steps):
    def __init__(self, moStressPreprocessing):
        self.moStressPreprocessing = moStressPreprocessing
        self._hasWeightsCalculationFinished = False

    def execute(self):
        try:
            if(not self._hasWeightsCalculationFinished):
                self.moStressPreprocessing.featuresModelValidation, self.moStressPreprocessing.targetsModelValidation =  self._getValidationData()
                self.moStressPreprocessing.features, self.moStressPreprocessing.targets = self._getTrainningData()
                self.moStressPreprocessing.weights = self._getWeights()
                self._hasWeightsCalculationFinished = True
        except:
            raise Exception("Weights Calculation failed.")

    def _getValidationData(self):
        seed(124816)
        indexToRemove = randint(0, self.moStressPreprocessing.quantityOfSets - 1)
        return self.moStressPreprocessing.features.pop(indexToRemove), self.moStressPreprocessing.targets.pop(indexToRemove)

    def _getTrainningData(self):
        trainningFeatures = self.moStressPreprocessing.features[0]
        trainningTargets = self.moStressPreprocessing.targets[0]
        for i in range(1, self.moStressPreprocessing.quantityOfSets - 1):
            trainningFeatures += self.moStressPreprocessing.features[i]
            trainningTargets += self.moStressPreprocessing.targets[i]
        return trainningFeatures, trainningTargets

    def _getWeights(self):
        weights = compute_class_weight(
            class_weight = "balanced", 
            classes = np.unique(self.moStressPreprocessing.targets),
            y = self.moStressPreprocessing.targets
        )
        weights = {i : weights[i] for i in range(len(weights))}
        return weights
