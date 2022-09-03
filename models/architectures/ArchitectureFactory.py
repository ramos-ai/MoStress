from models.architectures.SequentialArchitectures import \
    SequentialArchitectures


class ArchitectureFactory():
    def make(self, architectureName, architectureOptions):
        sequentialArchitecture = SequentialArchitectures(
            architectureOptions["winSize"], architectureOptions["numFeatures"], architectureOptions["numClasses"])
        if (architectureName == "REGULARIZER-GRU"):
            return sequentialArchitecture.gruRegularizerMoStress()
        elif (architectureName == "REGULARIZER-LSTM"):
            return sequentialArchitecture.lstmRegularizerMoStress()
        elif (architectureName == "BASELINE-GRU"):
            return sequentialArchitecture.gruBaselineMostress()
        elif (architectureName == "BASELINE-LSTM"):
            return sequentialArchitecture.lstmBaselineMostress()
        else:
            raise Exception("Architecture not implemented.")
