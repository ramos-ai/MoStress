from moStress.preprocessing.implementedSteps.Normalization import Normalization
from moStress.preprocessing.implementedSteps.WindowsLabelling import WindowsLabelling
from moStress.preprocessing.implementedSteps.WeightsCalculation import WeightsCalculation

class MoStressPreprocessing:
    def __init__(self, modelOpts, dataset) -> None:
        self.modelOpts = modelOpts
        self.quantityOfSets = len(dataset)

        self.datasetSamplePeriod = self.modelOpts["datasetSamplePeriod"]
        self.resamplingPeriod = self.modelOpts["resamplingPeriod"]
        self.winSize = self.modelOpts["windowSizeInSeconds"] * (self.datasetSamplePeriod // self.resamplingPeriod)
        self.winStep = self.modelOpts["windowSlideStepInSeconds"]

        self.features = [ data[::self.resamplingPeriod][ self.modelOpts["features"] ] for data in dataset ]
        self.targets = [ data[::self.resamplingPeriod][ self.modelOpts["targets"] ] for data in dataset ]
        self.winNumberBySubject = [ len(range(0, len(data) - self.winSize +  1, self.winStep)) for data in self.features ]

        self.countingThreshold = self.modelOpts["percentageCountingThreshold"] / 100
        self.targetsClassesMapping = self.modelOpts["targetsClassesMapping"]

        self.discartedWindosCounter = []

    def execute(self):
        print("Starting MoStress data preprocessing.\n")

        print("Starting Data Normalization.\n")
        self._applyNormalization()
        print("Normalization finished.\n")

        print("Starting Windows Labelling.\n")
        self._applyWindowsLabelling()
        print("Windows Labelling finished.\n")

        print("Starting Weights Calculation.\n")
        self._applyWeightsCaculation()
        print("Weights Calculation finished.\n")

        print("MoStress data preprocessing finished.\n")

        # TODO: implement Resume Report that might be printed in the ende



    def _applyNormalization(self):
        Normalization(self).execute()
    
    def _applyWindowsLabelling(self):
        WindowsLabelling(self).execute()
    
    def _applyWeightsCaculation(self):
        WeightsCalculation(self).execute()