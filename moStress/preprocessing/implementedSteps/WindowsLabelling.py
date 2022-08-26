from moStress.preprocessing.implementedSteps.Steps import Steps
import numpy as np

class WindowsLabelling(Steps):
    def __init__(self, moStressPreprocessing):
        self.moStressPreprocessing = moStressPreprocessing
        self._hasWindowsLabellingFinished = False
    
    def execute(self):
        try:
            if(not self._hasWindowsLabellingFinished):
                for i in range(self.moStressPreprocessing.quantityOfSets):
                    self.moStressPreprocessing.features[i], self.moStressPreprocessing.targets[i], discartedWindosCounter= self._labellingWindows(self.moStressPreprocessing.features[i], self.moStressPreprocessing.targets[i])
                    self.moStressPreprocessing.discartedWindosCounter.append(discartedWindosCounter)
                    self.moStressPreprocessing.updatedTargetsClassesMapping = {
                        str(int(key) - 1): self.moStressPreprocessing.targetsClassesMapping[key]
                        for key in self.moStressPreprocessing.targetsClassesMapping
                    }
                self._hasWindowsLabellingFinished = True
        except:
            raise Exception("Windows Labelling failled.")
    
    def _getElementArrayFrequency(self, npArray):
        (uniqueElements, elementsCount) = np.unique(npArray, return_counts=True, axis=0)

        mostFrequentElementIndexes = np.where(elementsCount == max(elementsCount))
        mostFrequentElementFirstIndex = mostFrequentElementIndexes[0][0]

        percentageFrequency = elementsCount / sum(elementsCount)

        return uniqueElements[mostFrequentElementFirstIndex], percentageFrequency[mostFrequentElementFirstIndex]
    
    def _labellingWindows(self, df, labels):

        featuresWindowsArray = []
        targetsLabelsArray = []
        discartedWindosCounter = { key: 0 for key in self.moStressPreprocessing.targetsClassesMapping }

        for i in range(len(df) - self.moStressPreprocessing._winSize + 1):
            slicer = slice(i, self.moStressPreprocessing._winSize + i, self.moStressPreprocessing._winStep)
            labelsNpArray = labels.to_numpy()[slicer]
            windowLabel, labelFrequency = self._getElementArrayFrequency(labelsNpArray)
        
            if (str(windowLabel) in self.moStressPreprocessing.targetsClassesMapping):
                if (labelFrequency >= self.moStressPreprocessing._countingThreshold):
                    featuresWindowsArray.append(df[slicer].to_numpy())
                    targetsLabelsArray.append(windowLabel - 1)
                else:
                    discartedWindosCounter[str(windowLabel)] += 1
        
        return featuresWindowsArray, targetsLabelsArray, discartedWindosCounter
    