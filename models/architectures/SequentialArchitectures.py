from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, LeakyReLU, GRU, GaussianNoise, Flatten, Activation
from tensorflow.python.keras.regularizers import L2
from tensorflow.python.keras.constraints import max_norm

class SequentialArchitectures():
    def __init__(self, winSize, numFeatures, numClasses):
        self.winSize = winSize
        self.numFeatures = numFeatures
        self.numClasses = numClasses
    
    def gruRegularizerMoStress(self):
        return self._rnnRegularizersMoStress(GRU, L2, 0.000001, max_norm, 3)

    def lstmRegularizerMoStress(self, ):
        return self._rnnRegularizersMoStress(LSTM, L2, 0.000001, max_norm, 3)

    def gruBaselineMostress(self, ):
        return self._rnnBaselineMoStress(GRU)

    def lstmBaselineMostress(self, ):
        return self._rnnBaselineMoStress(LSTM)

    def _rnnRegularizersMoStress(self, neuron, regularizerFunction, regularizationFactor, biasConstraintFunction,  biasConstraintFactor):
        regularizer = lambda regularizerFunc: regularizerFunc(regularizationFactor)
        biasConstraint = lambda biasFunc: biasFunc(biasConstraintFactor)
        return self._rnnBaselineMoStress(neuron, regularizer(regularizerFunction), biasConstraint(biasConstraintFunction))

    def _rnnBaselineMoStress(self, neuron, activity_regularizer = None, bias_constraint = None):
        rnnNeuron = lambda neuronFunc: neuronFunc(128, input_shape=(self.winSize, self.numFeatures), return_sequences=True)

        model = Sequential()
        model.add(rnnNeuron(neuron))
        model.add(LeakyReLU(alpha=0.5))
        model.add(GaussianNoise(0.5))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(self.numClasses, activity_regularizer = activity_regularizer, bias_constraint = bias_constraint))
        model.add(Activation("softmax"))
        model.summary()

        return model

if __name__ == "__main__":

    SequentialArchitectures(420, 5, 3).gruRegularizerMoStress()
    SequentialArchitectures(420, 5, 3).lstmRegularizerMoStress()
    SequentialArchitectures(420, 5, 3).gruBaselineMostress()
    SequentialArchitectures(420, 5, 3).lstmBaselineMostress()