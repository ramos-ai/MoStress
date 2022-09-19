from datasets.implementedDatasets.WesadPhysioChest import WesadPhysioChest
from datasets.implementedDatasets.StressInducingFeatures import StressInducingFeatures

class DatasetFactory:
    def make(self, datasetName, datasetPath, datasetOptions):
        if datasetName == "Wesad Physio Chest Data":
            return WesadPhysioChest(datasetPath, datasetOptions)
        if datasetName == "Stress-Inducing Features":
            return StressInducingFeatures(datasetPath, datasetOptions)
        else:
            raise Exception("Dataset not implemented.")
