from tensorflow.python import keras
import tensorflow.python.keras.backend as K

class TrendBlock(keras.layers.Layer):
    def __init__(self, 
                 lookback        = 7,
                 horizon         = 1,
                 num_neurons     = 512,
                 block_layers    = 4, 
                 polynomial_term = 2):
        super(TrendBlock, self).__init__()
        """Generic Block for Nbeats model.  Inputs:  
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon: int -> How far out into the future you would like
                             your predictions to be
            ----
            num_neurons: int -> How many layers to put into each Dense layer in
                                the generic block
            ----
            block_layers: int -> How many Dense layers to add to the block
            ----
            polynomial_term: int -> Degree of polynomial to use to understand
            trend term
            """
        self.polynomial_size = polynomial_term + 1
        self.layers_         = [keras.layers.Dense(num_neurons, 
                                                   activation = 'relu') 
                                for _ in range(block_layers)]
        self.lookback        = lookback
        self.horizon         = horizon
        self.theta_size      = 2 * (self.polynomial_size)
        self.backcast_size   = lookback * horizon
        self.theta           = keras.layers.Dense(self.theta_size, 
                                                  use_bias = False, 
                                                  activation = None)
        # taken from equation (2) in paper
        self.forecast_time   = K.concatenate([K.pow(K.arange(horizon, 
                                                             dtype = 'float') / horizon, i)[None, :]
                                 for i in range(self.polynomial_size)], axis = 0)
        self.backcast_time   = K.concatenate([K.pow(K.arange(self.backcast_size, 
                                                             dtype = 'float') / self.backcast_size, i)[None, :]
                                 for i in range(self.polynomial_size)], axis = 0)
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        x = self.theta(x)
        # create forecast / backcast from T / theta matrix
        backcast = K.dot(x[:, self.polynomial_size:], self.backcast_time)
        forecast = K.dot(x[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast