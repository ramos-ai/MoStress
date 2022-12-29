import sys
import os

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from os.path import join
import json
from moStress.neuralNetwork.MoStressNeuralNetwork import MoStressNeuralNetwork
from utils.preprocessingCheckpoint import getPreprocessingCheckpoint
from models.EvaluateModel import EvaluateModel
from datasets.Dataset import Dataset
from models.architectures.NBeatsFeatureExtractor import TIME_SERIES_TO_PROCESS, NBeatsFeatureExtractor

sys.stdout = open(join("main", "04-nbeatsFeatureExtractor", "stdout.txt"), "a")
sys.stderr = open(join("main", "04-nbeatsFeatureExtractor", "stderr.txt"), "a")

MODEL_T0_TEST = "NBEATS-FEATURE-EXTRACTOR"

moStressJsonFilePath = join("configs", "moStressConfigs.json")

with open(moStressJsonFilePath, "r") as j:
    moStressConfigs = json.loads(j.read())

trainData, validationData = getPreprocessingCheckpoint()

dataset = {
    "features": trainData["features"],
    "targets": trainData["targets"],
    "weights": trainData["weights"],
}

moStressNeuralNetwork = MoStressNeuralNetwork(moStressConfigs, dataset, True)

moStressNeuralNetwork.execute(MODEL_T0_TEST, "adam", "sequential")

NBeatsFeatureExtractor.collectValidationResiduals(validationData["features"])
validationData["features"] = Dataset.loadData(
    join(
        "data",
        "preprocessedData",
        "residuals",
        "validation",
        f"residualTimeSeries_{TIME_SERIES_TO_PROCESS}.pickle"
    )
)

evaluator = EvaluateModel(
    {"features": validationData["features"], "targets": validationData["targets"]},
    MODEL_T0_TEST,
    moStressNeuralNetwork.model,
)

evaluator.executeEvaluation()
