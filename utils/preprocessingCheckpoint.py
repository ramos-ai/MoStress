import os
from datasets.Dataset import Dataset

saveDataPath = "data/preprocessedData"

def setPreprocessingCheckpoint(moStressPreprocessing):

    Dataset.saveData(
        saveDataPath + "/trainning/trainningData.pickle",
        {
            "features": moStressPreprocessing.features,
            "targets": moStressPreprocessing.targets,
            "weights": moStressPreprocessing.weights
        }
    )

    Dataset.saveData(
        saveDataPath + "/validation/validationData.pickle",
        {
            "features": moStressPreprocessing.featuresModelValidation,
            "targets": moStressPreprocessing.targetsModelValidation,
        }
    )

def getPreprocessingCheckpoint():
    if (not os.listdir(saveDataPath)):
        raise Exception("You need to set the checkpoint first.")
    return Dataset.loadData("data/preprocessedData/trainning/trainningData.pickle")
