from models.architectures.ReservoirModels import ReservoirModels
from models.architectures.SequentialModels import \
    SequentialModels


class ModelFactory():
    def make(self, architectureName, architectureOptions):
        sequentialArchitecture = SequentialModels(
            architectureOptions["winSize"], architectureOptions["numFeatures"], architectureOptions["numClasses"])
        if (architectureName == "REGULARIZER-GRU"):
            return sequentialArchitecture.gruRegularizerMoStress()
        elif (architectureName == "REGULARIZER-LSTM"):
            return sequentialArchitecture.lstmRegularizerMoStress()
        elif (architectureName == "BASELINE-GRU"):
            return sequentialArchitecture.gruBaselineMostress()
        elif (architectureName == "BASELINE-LSTM"):
            return sequentialArchitecture.lstmBaselineMostress()
        elif (architectureName == "BASELINE-RESERVOIR"):
            return ReservoirModels().baseline()
        else:
            raise Exception("Architecture not implemented.")
