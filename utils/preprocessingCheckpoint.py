import os

from datasets.Dataset import Dataset

saveDataPathRoot = os.path.join("data", "preprocessedData")
trainingDataPath = os.path.join(
    saveDataPathRoot, "training", "trainingData.pickle")
validationDataPath = os.path.join(
    saveDataPathRoot, "validation", "validationData.pickle"),


def setPreprocessingCheckpoint(moStressPreprocessing):

    Dataset.saveData(
        trainingDataPath,
        {
            "features": moStressPreprocessing.features,
            "targets": moStressPreprocessing.targets,
            "weights": moStressPreprocessing.weights
        }
    )

    Dataset.saveData(
        validationDataPath,
        {
            "features": moStressPreprocessing.featuresModelValidation,
            "targets": moStressPreprocessing.targetsModelValidation,
        }
    )


def getPreprocessingCheckpoint():
    if (not os.listdir(saveDataPathRoot)):
        raise Exception("You need to set the checkpoint first.")
    return Dataset.loadData(trainingDataPath), Dataset.loadData(validationDataPath)
