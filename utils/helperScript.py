import sys
import os

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from datasets.Dataset import Dataset
from utils.utils import deleteData, createFolder, hasDataFile
from sklearn.model_selection import train_test_split
from utils.Logger import LogLevel, Logger

logInfo = Logger("ResidualManager", LogLevel.INFO)

OLD_BATCHES = [1, 2, 3]
NEW_BATCHES = [1, 2, 3, 4]
TIME_SERIES_INDEX = [0, 1, 2, 3, 4]
basePathForNewResiduals = os.path.join("data","preprocessedData","residuals")

checkPoint = Dataset.loadData("data/preprocessedData/training/trainingData.pickle")
y = checkPoint["targets"][:-1]
deleteData(checkPoint)

def joinOldBatches(timeSeriesIndex):
    path = f"data/preprocessedData/residuals/training/residualTimeSeries_{timeSeriesIndex}_4.pickle"
    if (hasDataFile(path)):
        logInfo("Old batches jointed already.")
        return
    x = []
    for i in OLD_BATCHES:
        x += Dataset.loadData(f"data/preprocessedData/residuals/training/residualTimeSeries_{timeSeriesIndex}_{i}.pickle")
    Dataset.saveData(path, x)
    deleteData(x)

def createNewBatches(timeSeriesIndex):
    timeSeriesFolderPath = os.path.join(basePathForNewResiduals, f"timeSeries{timeSeriesIndex}")
    createFolder(timeSeriesFolderPath)
    for i in NEW_BATCHES:
        batchPath = os.path.join(timeSeriesFolderPath, f"batch{i}")
        createFolder(batchPath)

    x = Dataset.loadData(f"data/preprocessedData/residuals/training/residualTimeSeries_{timeSeriesIndex}_4.pickle")

    logInfo("Getting half.")
    xHalf1, xHalf2, yHalf1, yHalf2 = train_test_split(x, y, test_size=0.5, stratify=y)
    deleteData(x)
    logInfo("Half collected.")

    logInfo("Getting first quarters.")
    xBatch1, xBatch2, yBatch1, yBatch2 = train_test_split(xHalf1, yHalf1, test_size=0.5, stratify=yHalf1)
    logInfo("Quarters collected.")

    deleteData(xHalf1)
    deleteData(yHalf1)

    logInfo("Saving quarters collected.")
    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch1", "features.pickle"), xBatch1)
    deleteData(xBatch1)

    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch1", "targets.pickle"), yBatch1)
    deleteData(yBatch1)

    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch2", "features.pickle"), xBatch2)
    deleteData(xBatch2)

    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch2", "targets.pickle"), yBatch2)
    deleteData(yBatch2)
    logInfo("Quarters saved.")

    logInfo("Getting second quarters.")
    xBatch3, xBatch4, yBatch3, yBatch4 = train_test_split(xHalf2, yHalf2, test_size=0.5, stratify=yHalf2)
    logInfo("Quarters collected.")

    deleteData(xHalf2)
    deleteData(yHalf2)

    logInfo("Saving quarters collected.")
    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch3", "features.pickle"), xBatch3)
    deleteData(xBatch3)

    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch3", "targets.pickle"), yBatch3)
    deleteData(yBatch3)

    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch4", "features.pickle"), xBatch4)
    deleteData(xBatch4)

    Dataset.saveData(os.path.join(timeSeriesFolderPath, "batch4", "targets.pickle"), yBatch4)
    deleteData(yBatch4)
    logInfo("Quarters saved.")

def main():
    for i in TIME_SERIES_INDEX:
        logInfo(f"Starting to manage residuals of Time Series {i}.")
        logInfo(f"Starting to join old batches.")
        joinOldBatches(i)
        logInfo(f"Starting to create new batches.")
        createNewBatches(i)
        logInfo(f"Time Series {i} finished, moving to the next one.")
    logInfo("Done")


main()
