import os

from datasets.Dataset import Dataset

saveDataPathRoot = os.path.join("..","data", "preprocessedData")

def datasetDataFiles(dataset):
    trainingDataPath = os.path.join(
        saveDataPathRoot, "training", f"trainingData{dataset}.pickle")
    validationDataPath = os.path.join(
        saveDataPathRoot, "validation", f"validationData{dataset}.pickle")
    return trainingDataPath, validationDataPath

def checkPreprocessingCheckpoint():
    os.makedirs(f"{saveDataPathRoot}/training", exist_ok=True)
    os.makedirs(f"{saveDataPathRoot}/validation", exist_ok=True)

def setPreprocessingCheckpoint(moStressPreprocessing, dataset):
    checkPreprocessingCheckpoint()
    trainingDataPath, validationDataPath = datasetDataFiles(dataset)

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


def getPreprocessingCheckpoint(dataset):
    if (not os.listdir(saveDataPathRoot)):
        raise Exception("You need to set the checkpoint first.")
    trainingDataPath, validationDataPath = datasetDataFiles(dataset)
    return Dataset.loadData(trainingDataPath), Dataset.loadData(validationDataPath)

