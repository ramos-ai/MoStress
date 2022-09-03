from datasets.implementedDatasets.WesadPhysioChest import WesadPhysioChest


class DatasetFactory:
    def make(self, datasetName, datasetPath, datasetOptions):
        if datasetName == "Wesad Physio Chest Data":
            return WesadPhysioChest(datasetPath, datasetOptions)
        else:
            raise Exception("Dataset not implemented.")
