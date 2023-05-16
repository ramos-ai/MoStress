from Logger import LogLevel, Logger
from sklearn.model_selection import train_test_split
from utils import deleteData, createFolder, hasDataFile
from datasets.Dataset import Dataset
import sys
import os
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.environ["VIRTUAL_ENV"])
sys.path.append(ROOT_DIR)


logInfo = Logger("ResidualManager", LogLevel.INFO)

OLD_BATCHES = [1, 2, 3]
NEW_BATCHES = [1, 2, 3, 4]
JOINT_TIME_SERIES_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8]
TIME_SERIES_INDEX = [0, 1, 2, 3, 4]
basePathForNewResiduals = os.path.join("data", "preprocessedData", "residuals")

checkPoint = Dataset.loadData(
    "data/preprocessedData/training/trainingData.pickle")
y = checkPoint["targets"][:-1]
deleteData(checkPoint)


def joinOldBatches(timeSeriesIndex):
    path = f"data/preprocessedData/residuals/training/residualTimeSeries_{timeSeriesIndex}_4.pickle"
    if (hasDataFile(path)):
        logInfo("Old batches jointed already.")
        return
    x = []
    for i in OLD_BATCHES:
        x += Dataset.loadData(
            f"data/preprocessedData/residuals/training/residualTimeSeries_{timeSeriesIndex}_{i}.pickle")
    Dataset.saveData(path, x)
    deleteData(x)


def createNewBatches(timeSeriesIndex):
    timeSeriesFolderPath = os.path.join(
        basePathForNewResiduals, f"timeSeries{timeSeriesIndex}")
    createFolder(timeSeriesFolderPath)
    for i in NEW_BATCHES:
        batchPath = os.path.join(timeSeriesFolderPath, f"batch{i}")
        createFolder(batchPath)

    x = Dataset.loadData(
        f"data/preprocessedData/residuals/training/residualTimeSeries_{timeSeriesIndex}_4.pickle")

    logInfo("Getting half.")
    xHalf1, xHalf2, yHalf1, yHalf2 = train_test_split(
        x, y, test_size=0.5, stratify=y)
    deleteData(x)
    logInfo("Half collected.")

    logInfo("Getting first quarters.")
    xBatch1, xBatch2, yBatch1, yBatch2 = train_test_split(
        xHalf1, yHalf1, test_size=0.5, stratify=yHalf1)
    logInfo("Quarters collected.")

    deleteData(xHalf1)
    deleteData(yHalf1)

    logInfo("Saving quarters collected.")
    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch1", "features.pickle"), xBatch1)
    deleteData(xBatch1)

    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch1", "targets.pickle"), yBatch1)
    deleteData(yBatch1)

    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch2", "features.pickle"), xBatch2)
    deleteData(xBatch2)

    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch2", "targets.pickle"), yBatch2)
    deleteData(yBatch2)
    logInfo("Quarters saved.")

    logInfo("Getting second quarters.")
    xBatch3, xBatch4, yBatch3, yBatch4 = train_test_split(
        xHalf2, yHalf2, test_size=0.5, stratify=yHalf2)
    logInfo("Quarters collected.")

    deleteData(xHalf2)
    deleteData(yHalf2)

    logInfo("Saving quarters collected.")
    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch3", "features.pickle"), xBatch3)
    deleteData(xBatch3)

    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch3", "targets.pickle"), yBatch3)
    deleteData(yBatch3)

    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch4", "features.pickle"), xBatch4)
    deleteData(xBatch4)

    Dataset.saveData(os.path.join(timeSeriesFolderPath,
                     "batch4", "targets.pickle"), yBatch4)
    deleteData(yBatch4)
    logInfo("Quarters saved.")


def generateNewResidualsParts(basePath, timeSeriesIndex, inputBatchIndex, outputBatchIndex):
    logInfo("Loading Data")
    x1 = Dataset.loadData(
        f"data/preprocessedData/residuals/timeSeries{timeSeriesIndex}/batch{inputBatchIndex}/features.pickle")
    y1 = Dataset.loadData(
        f"data/preprocessedData/residuals/timeSeries{timeSeriesIndex}/batch{inputBatchIndex}/targets.pickle")
    logInfo("Data Loaded")

    logInfo("Splitting Data")
    x1Half1, x1Half2, y1Half1, y1Half2 = train_test_split(
        x1, y1, test_size=0.5, stratify=y1)
    logInfo("Data Splitted")
    deleteData(x1)
    deleteData(y1)

    logInfo("Saving Data")
    Dataset.saveData(os.path.join(
        basePath, f"batch{outputBatchIndex}", f"temp_feature_{timeSeriesIndex}.pickle"), x1Half1)
    deleteData(x1Half1)
    Dataset.saveData(os.path.join(
        basePath, f"batch{outputBatchIndex + 1}", f"temp_feature_{timeSeriesIndex}.pickle"), x1Half2)
    deleteData(y1Half1)
    Dataset.saveData(os.path.join(
        basePath, f"batch{outputBatchIndex}", "targets.pickle"), y1Half1)
    deleteData(y1Half1)
    Dataset.saveData(os.path.join(
        basePath, f"batch{outputBatchIndex + 1}", "targets.pickle"), y1Half2)
    deleteData(y1Half2)
    logInfo("Data Saved")


def createPartialJointTimeSeriesResidual(inputTimeSeriesIndex):

    jointTimeSeriesPath = os.path.join(basePathForNewResiduals, "timeSeries5")
    createFolder(jointTimeSeriesPath)

    for i in JOINT_TIME_SERIES_BATCHES:
        createFolder(os.path.join(jointTimeSeriesPath, f"batch{i+1}"))

    logInfo("Generating new batches 1 and 2")
    generateNewResidualsParts(
        jointTimeSeriesPath, timeSeriesIndex=inputTimeSeriesIndex, inputBatchIndex=1, outputBatchIndex=1)
    logInfo("Generating new batches 3 and 4")
    generateNewResidualsParts(
        jointTimeSeriesPath, timeSeriesIndex=inputTimeSeriesIndex, inputBatchIndex=2, outputBatchIndex=3)
    logInfo("Generating new batches 5 and 6")
    generateNewResidualsParts(
        jointTimeSeriesPath, timeSeriesIndex=inputTimeSeriesIndex, inputBatchIndex=3, outputBatchIndex=5)
    logInfo("Generating new batches 7 and 8")
    generateNewResidualsParts(
        jointTimeSeriesPath, timeSeriesIndex=inputTimeSeriesIndex, inputBatchIndex=4, outputBatchIndex=7)
    logInfo("New Batches Generated")


def joinPartialResiduals(jointTimeSeriesBatchIndex):
    logInfo("Collecting All Partials")
    allPartialResiduals = [Dataset.loadData(os.path.join(basePathForNewResiduals, "timeSeries5",
                                            f"batch{jointTimeSeriesBatchIndex}", f"temp_feature_{timeSeriesIndex}.pickle")) for timeSeriesIndex in TIME_SERIES_INDEX]
    logInfo("Converting to Tensor..... Tense moment...")
    tensor = tf.convert_to_tensor(allPartialResiduals)
    deleteData(allPartialResiduals)
    logInfo("Transposing the Tensor")
    transposedTensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    deleteData(tensor)
    logInfo("Saving the Tensor")
    Dataset.saveData(os.path.join(basePathForNewResiduals, f"batch{jointTimeSeriesBatchIndex}", "features.pickle"), transposedTensor.numpy().tolist())
    deleteData(transposedTensor)
    logInfo("Tensor Saved")



def main():
    logInfo("Start to create Joint Time Series Data")
    for i in TIME_SERIES_INDEX:
        logInfo(f"Starting with Time Series {i}")
        createPartialJointTimeSeriesResidual(i)
    for i in JOINT_TIME_SERIES_BATCHES:
        logInfo(f"Starting to join partials of the new batch {i}")
        joinPartialResiduals(i)
    logInfo("Done")


main()
