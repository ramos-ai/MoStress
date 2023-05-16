import sys
import os

ROOT_DIR = os.path.dirname(os.environ["VIRTUAL_ENV"])
sys.path.append(ROOT_DIR)

from Logger import LogLevel, Logger
from sklearn.model_selection import train_test_split
from utils import deleteData, createFolder, hasDataFile
from datasets.Dataset import Dataset
import tensorflow as tf
import shutil
import numpy as np


logInfo = Logger("ResidualManager", LogLevel.INFO)

OLD_BATCHES = [1, 2, 3]
NEW_BATCHES = [1, 2, 3, 4]
JOINT_TIME_SERIES_BATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
TIME_SERIES_INDEX = [0, 1, 2, 3, 4]
BEST_TIME_SERIES_INDEX = [2, 4]
NEW_TIME_SERIES_INDEX = 6
basePathForNewResiduals = os.path.join("data", "preprocessedData", "residuals")
auxBasePathForNewResiduals = os.path.join("data", "preprocessedData", "residuals2")
binaryClassificationFolder = os.path.join("data", "preprocessedData", "residualsBinary")
createFolder(basePathForNewResiduals)
createFolder(auxBasePathForNewResiduals)
createFolder(binaryClassificationFolder)

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
    for i in JOINT_TIME_SERIES_BATCHES:
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
    xQuarter1, xQuarter2, yQuarter1, yQuarter2 = train_test_split(
        xHalf1, yHalf1, test_size=0.5, stratify=yHalf1)
    logInfo("Quarters collected.")

    deleteData(xHalf1)
    deleteData(yHalf1)

    logInfo("Getting first eighths.")
    xBatch1, xBatch2, yBatch1, yBatch2 = train_test_split(
        xQuarter1, yQuarter1, test_size=0.5, stratify=yQuarter1)
    logInfo("Eighths collected.")

    deleteData(xQuarter1)
    deleteData(yQuarter1)

    logInfo("Saving eights collected.")
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
    logInfo("Eighths saved.")

    logInfo("Getting second eighths.")
    xBatch3, xBatch4, yBatch3, yBatch4 = train_test_split(
        xQuarter2, yQuarter2, test_size=0.5, stratify=yQuarter2)
    logInfo("Eighths collected.")

    deleteData(xQuarter2)
    deleteData(yQuarter2)

    logInfo("Saving eights collected.")
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
    logInfo("Eighths saved.")

    logInfo("Getting second quarters.")
    xQuarter3, xQuarter4, yQuarter3, yQuarter4 = train_test_split(
        xHalf2, yHalf2, test_size=0.5, stratify=yHalf2)
    logInfo("Quarters collected.")

    logInfo("Getting third eighths.")
    xBatch5, xBatch6, yBatch5, yBatch6 = train_test_split(
        xQuarter3, yQuarter3, test_size=0.5, stratify=yQuarter3)
    logInfo("Eighths collected.")

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

def splitDataInHalf(x, y):
    xHalf1, xHalf2, yHalf1, yHalf2 = train_test_split(x, y, test_size=0.5, stratify=y)
    deleteData(x)
    deleteData(y)
    return xHalf1, xHalf2, yHalf1, yHalf2

def saveSplits(basePath, batchIndex, x, y):
    Dataset.saveData(os.path.join(basePath,
                     f"batch{batchIndex}", "features.pickle"), x)
    deleteData(x)
    Dataset.saveData(os.path.join(basePath,
                     f"batch{batchIndex}", "targets.pickle"), y)
    deleteData(y)

def splitDataInChunks(timeSeriesIndex, x, y):

    timeSeriesFolderPath = os.path.join(
        basePathForNewResiduals, f"timeSeries{timeSeriesIndex}")
    
    for i in JOINT_TIME_SERIES_BATCHES:
        batchPath = os.path.join(timeSeriesFolderPath, f"batch{i}")
        createFolder(batchPath)

    xHalf1, xHalf2, yHalf1, yHalf2 = splitDataInHalf(x, y)
    
    xQuarter1, xQuarter2, yQuarter1, yQuarter2 = splitDataInHalf(xHalf1, yHalf1)
    xQuarter3, xQuarter4, yQuarter3, yQuarter4 = splitDataInHalf(xHalf2, yHalf2)

    xEighths1, xEighths2, yEighths1, yEighths2 = splitDataInHalf(xQuarter1, yQuarter1)
    xEighths3, xEighths4, yEighths3, yEighths4 = splitDataInHalf(xQuarter2, yQuarter2)
    xEighths5, xEighths6, yEighths5, yEighths6 = splitDataInHalf(xQuarter3, yQuarter3)
    xEighths7, xEighths8, yEighths7, yEighths8 = splitDataInHalf(xQuarter4, yQuarter4)

    xSixteenth1, xSixteenth2, ySixteenth1, ySixteenth2 = splitDataInHalf(xEighths1, yEighths1)
    xSixteenth3, xSixteenth4, ySixteenth3, ySixteenth4 = splitDataInHalf(xEighths2, yEighths2)
    xSixteenth5, xSixteenth6, ySixteenth5, ySixteenth6 = splitDataInHalf(xEighths3, yEighths3)
    xSixteenth7, xSixteenth8, ySixteenth7, ySixteenth8 = splitDataInHalf(xEighths4, yEighths4)
    xSixteenth9, xSixteenth10, ySixteenth9, ySixteenth10 = splitDataInHalf(xEighths5, yEighths5)
    xSixteenth11, xSixteenth12, ySixteenth11, ySixteenth12 = splitDataInHalf(xEighths6, yEighths6)
    xSixteenth13, xSixteenth14, ySixteenth13, ySixteenth14 = splitDataInHalf(xEighths7, yEighths7)
    xSixteenth15, xSixteenth16, ySixteenth15, ySixteenth16 = splitDataInHalf(xEighths8, yEighths8)

    saveSplits(timeSeriesFolderPath, 1, xSixteenth1, ySixteenth1)
    saveSplits(timeSeriesFolderPath, 2, xSixteenth2, ySixteenth2)
    saveSplits(timeSeriesFolderPath, 3, xSixteenth3, ySixteenth3)
    saveSplits(timeSeriesFolderPath, 4, xSixteenth4, ySixteenth4)
    saveSplits(timeSeriesFolderPath, 5, xSixteenth5, ySixteenth5)
    saveSplits(timeSeriesFolderPath, 6, xSixteenth6, ySixteenth6)
    saveSplits(timeSeriesFolderPath, 7, xSixteenth7, ySixteenth7)
    saveSplits(timeSeriesFolderPath, 8, xSixteenth8, ySixteenth8)
    saveSplits(timeSeriesFolderPath, 9, xSixteenth9, ySixteenth9)
    saveSplits(timeSeriesFolderPath, 10, xSixteenth10, ySixteenth10)
    saveSplits(timeSeriesFolderPath, 11, xSixteenth11, ySixteenth11)
    saveSplits(timeSeriesFolderPath, 12, xSixteenth12, ySixteenth12)
    saveSplits(timeSeriesFolderPath, 13, xSixteenth13, ySixteenth13)
    saveSplits(timeSeriesFolderPath, 14, xSixteenth14, ySixteenth14)
    saveSplits(timeSeriesFolderPath, 15, xSixteenth15, ySixteenth15)
    saveSplits(timeSeriesFolderPath, 16, xSixteenth16, ySixteenth16)

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

    jointTimeSeriesPath = os.path.join(basePathForNewResiduals, f"timeSeries{NEW_TIME_SERIES_INDEX}")
    createFolder(jointTimeSeriesPath)

    for i in JOINT_TIME_SERIES_BATCHES:
        createFolder(os.path.join(jointTimeSeriesPath, f"batch{i}"))

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
    logInfo("Generating new batches 9 and 10")
    generateNewResidualsParts(
        jointTimeSeriesPath, timeSeriesIndex=inputTimeSeriesIndex, inputBatchIndex=4, outputBatchIndex=7)
    logInfo("Generating new batches 9 and 10")
    generateNewResidualsParts(
        jointTimeSeriesPath, timeSeriesIndex=inputTimeSeriesIndex, inputBatchIndex=4, outputBatchIndex=7)
    logInfo("New Batches Generated")

def joinPartialResiduals(jointTimeSeriesBatchIndex):
    logInfo("Collecting All Partials")
    allPartialResiduals = [Dataset.loadData(os.path.join(basePathForNewResiduals, f"timeSeries{NEW_TIME_SERIES_INDEX}",
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

def getJointTimeSeriesResidual(batchIndex):
    logInfo("Collecting All Partials")
    allPartialResiduals = [Dataset.loadData(os.path.join(auxBasePathForNewResiduals, f"timeSeries{timeSeriesIndex}",
                                            f"batch{batchIndex}", f"features.pickle")) for timeSeriesIndex in TIME_SERIES_INDEX]
    logInfo("Converting to Tensor..... Tense moment...")
    tensor = tf.convert_to_tensor(allPartialResiduals)
    deleteData(allPartialResiduals)
    logInfo("Transposing the Tensor")
    transposedTensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    deleteData(tensor)
    logInfo("Saving the Tensor")
    Dataset.saveData(os.path.join(basePathForNewResiduals, f"timeSeries{NEW_TIME_SERIES_INDEX}", f"batch{batchIndex}", "features.pickle"), transposedTensor.numpy().tolist())
    deleteData(transposedTensor)
    logInfo("Tensor Saved")

def copingTargets(batchIndex):
    for timeSeriesIndex in TIME_SERIES_INDEX:
        logInfo(f"Loading Targets from Time Series {timeSeriesIndex}")
        y = Dataset.loadData(os.path.join(auxBasePathForNewResiduals, f"timeSeries{timeSeriesIndex}", f"batch{batchIndex}", "targets.pickle"))
        Dataset.saveData(os.path.join(basePathForNewResiduals, f"timeSeries{NEW_TIME_SERIES_INDEX}", f"batch{batchIndex}", "targets.pickle"), y)
        logInfo(f"Targets from Time Series {timeSeriesIndex} saved.")

def adjustValidationData():
    logInfo("Collecting All Partials")
    allPartialResiduals = [Dataset.loadData(os.path.join(basePathForNewResiduals, "validation", f"residualTimeSeries_{timeSeriesIndex}.pickle")) for timeSeriesIndex in TIME_SERIES_INDEX]
    logInfo("Converting to Tensor..... Tense moment...")
    tensor = tf.convert_to_tensor(allPartialResiduals)
    deleteData(allPartialResiduals)
    logInfo("Transposing the Tensor")
    transposedTensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
    deleteData(tensor)
    logInfo("Saving the Tensor")
    Dataset.saveData(os.path.join(basePathForNewResiduals, "validation", f"residualTimeSeries_{NEW_TIME_SERIES_INDEX}.pickle"), transposedTensor.numpy().tolist())
    deleteData(transposedTensor)
    logInfo("Tensor Saved")

def adjustTargetsForBinaryClassification(batchIndex):
    logInfo("Loading Targets")
    targets = Dataset.loadData(os.path.join(basePathForNewResiduals, f"timeSeries{NEW_TIME_SERIES_INDEX}", f"batch{batchIndex}", "targets.pickle"))
    logInfo("Loaded")
    logInfo("Transforming")
    npTargets = np.array(targets)
    npTargets[np.where(npTargets != 1)] = 0
    newTargets = npTargets.tolist()
    logInfo("Transformed")
    logInfo("Saving")
    Dataset.saveData(
        os.path.join(binaryClassificationFolder, f"timeSeries{NEW_TIME_SERIES_INDEX}", f"batch{batchIndex}", "targets.pickle"),
        newTargets
    )
    logInfo("Saved")

def main():
    createFolder(os.path.join(binaryClassificationFolder, f"timeSeries{NEW_TIME_SERIES_INDEX}"))
    for i in JOINT_TIME_SERIES_BATCHES:
        batchFolder = os.path.join(basePathForNewResiduals, f"timeSeries{NEW_TIME_SERIES_INDEX}", f"batch{i}")
        batchBinaryFolder = os.path.join(binaryClassificationFolder, f"timeSeries{NEW_TIME_SERIES_INDEX}", f"batch{i}")
        createFolder(batchBinaryFolder)
        logInfo(f"Copying Feature of Batch {i}")
        shutil.copy2(
            os.path.join(batchFolder, "features.pickle"),
            batchBinaryFolder
        )
        logInfo(f"Copy Finished!")
        logInfo(f"Starting Adjust of Targets")
        adjustTargetsForBinaryClassification(i)
    logInfo(f"All Done!")

main()
