import sys
import os

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

from os.path import join
import json
from moStress.neuralNetwork.MoStressNeuralNetwork import MoStressNeuralNetwork
from utils.preprocessingCheckpoint import getPreprocessingCheckpoint
from models.EvaluateModel import EvaluateModel
from utils.Logger import Logger, LogLevel

MODEL_T0_TEST = "NBEATS-FEATURE-EXTRACTOR"

moStressJsonFilePath = join("configs","moStressConfigs.json")

with open(moStressJsonFilePath, 'r') as j:
        moStressConfigs = json.loads(j.read())

trainData, validationData = getPreprocessingCheckpoint()

dataset = {
    "features": trainData["features"][:4],
    "targets": trainData["targets"][:4],
    "weights": trainData["weights"],
}

moStressNeuralNetwork = MoStressNeuralNetwork(moStressConfigs, dataset, True)

moStressNeuralNetwork.execute(
    MODEL_T0_TEST,
    "adam",
    "sequential"
)

evaluator = EvaluateModel(
    { "features": validationData["features"], "targets": validationData["targets"] },
    MODEL_T0_TEST,
    moStressNeuralNetwork.model,
)

evaluator.executeEvaluation()