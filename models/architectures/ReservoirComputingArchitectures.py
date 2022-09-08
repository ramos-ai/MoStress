from reservoirpy import set_seed, verbosity
from reservoirpy.nodes import Reservoir, Ridge, Input

class ReservoirComputingArchitectures:
    def __init__(self, randomSeed = 42, verbosityState = 0):
        set_seed(randomSeed)
        verbosity(verbosityState)

    def baseline(self, numNodes = 128, spectralRatio = 0.9, leakingRate = 0.1, ridgeFactor = 1e-6):
        source = Input()
        reservoir = Reservoir(numNodes, sr=spectralRatio, lr=leakingRate)
        readout = Ridge(ridge=ridgeFactor)

        model = source >> reservoir >> readout

        return {
            "model": model,
            "input": source,
            "reservoir": reservoir,
            "output": readout
        }

if __name__ == "__main__":
    ReservoirComputingArchitectures().baseline()
