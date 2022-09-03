from moStress.preprocessing.implementedSteps.Normalization import Normalization
from moStress.preprocessing.implementedSteps.WindowsLabelling import WindowsLabelling
from moStress.preprocessing.implementedSteps.WeightsCalculation import WeightsCalculation

class MoStressPreprocessing:
    def __init__(self, modelOpts, dataset) -> None:
        self.modelOpts = modelOpts
        self.quantityOfSets = len(dataset)

        self._datasetSamplePeriod = self.modelOpts["datasetSamplePeriod"]
        self._resamplingPeriod = self.modelOpts["resamplingPeriod"]
        self._winSize = self.modelOpts["windowSizeInSeconds"] * (self._datasetSamplePeriod // self._resamplingPeriod)
        self._winStep = self.modelOpts["windowSlideStepInSeconds"]

        self.features = [ data[::self._resamplingPeriod][ self.modelOpts["features"] ] for data in dataset ]
        self.targets = [ data[::self._resamplingPeriod][ self.modelOpts["targets"] ] for data in dataset ]
        self._winNumberBySubject = [ len(range(0, len(data) - self._winSize +  1, self._winStep)) for data in self.features ]

        self._countingThreshold = self.modelOpts["percentageCountingThreshold"] / 100
        self.targetsClassesMapping = self.modelOpts["targetsClassesMapping"]

        self.discardedWindowsCounter = []

    def execute(self):
        print("Starting MoStress data preprocessing.\n")

        print("Data Normalization in progress...\n")
        self._applyNormalization()
        print("Normalization finished.\n")

        print("Windows Labelling in progress...\n")
        self._applyWindowsLabelling()
        print("Windows Labelling finished.\n")

        print("Weights Calculation in progress...\n")
        self._applyWeightsCalculation()
        print("Weights Calculation finished.\n")

        print("MoStress data preprocessing finished.\n")

        # TODO: implement Resume Report that might be printed in the end

    def _applyNormalization(self):
        Normalization(self).execute()
    
    def _applyWindowsLabelling(self):
        WindowsLabelling(self).execute()
    
    def _applyWeightsCalculation(self):
        WeightsCalculation(self).execute()