from models.architectures.NBeatsFeatureExtractor import NBeatsFeatureExtractor
from models.architectures.ReservoirModels import ReservoirModels
from models.architectures.SequentialModels import SequentialModels


class ModelFactory:
    def make(self, moStressNeuralNetwork, modelName):
        if modelName == "REGULARIZER-GRU":
            return SequentialModels(moStressNeuralNetwork).gruRegularizerMoStress()
        elif modelName == "REGULARIZER-LSTM":
            return SequentialModels(moStressNeuralNetwork).lstmRegularizerMoStress()
        elif modelName == "BASELINE-GRU":
            return SequentialModels(moStressNeuralNetwork).gruBaselineMostress()
        elif modelName == "BASELINE-LSTM":
            return SequentialModels(moStressNeuralNetwork).lstmBaselineMostress()
        elif modelName == "BASELINE-RESERVOIR":
            return ReservoirModels(moStressNeuralNetwork).baseline()
        elif modelName == "NBEATS-FEATURE-EXTRACTOR":
            return NBeatsFeatureExtractor(moStressNeuralNetwork).classificationModel()
        else:
            raise Exception("Architecture not implemented.")
