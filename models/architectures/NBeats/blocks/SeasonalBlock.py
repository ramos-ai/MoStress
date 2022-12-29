import numpy as np
from tensorflow.python import keras
import tensorflow.python.keras.backend as K

class SeasonalBlock(keras.layers.Layer):
    def __init__(self, 
                 lookback      = 7,
                 horizon       = 1,
                 num_neurons   = 512,
                 block_layers  = 4,
                 num_harmonics = 1):
        super(SeasonalBlock, self).__init__()
        """Seasonality Block for Nbeats model.  Inputs:  
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
            num_harmonics: int -> The seasonal lag to use for your training window
        """
        self.layers_       = [keras.layers.Dense(num_neurons, 
                                                 activation = 'relu') 
                              for _ in range(block_layers)]
        self.lookback      = lookback
        self.horizon       = horizon
        self.num_harmonics = num_harmonics
        self.theta_size    = 4 * int(np.ceil(num_harmonics / 2 * horizon) - (num_harmonics - 1))
        self.backcast_size = lookback * horizon
        self.theta         = keras.layers.Dense(self.theta_size, 
                                                use_bias = False, 
                                                activation = None)
        self.frequency     = K.concatenate((K.zeros(1, dtype = 'float'), 
                             K.arange(num_harmonics, num_harmonics / 2 * horizon) / num_harmonics), 
                             axis = 0)

        self.backcast_grid = -2 * np.pi * (K.arange(self.backcast_size, dtype = 'float')[:, None] / self.backcast_size) * self.frequency

        self.forecast_grid = 2 * np.pi * (K.arange(horizon, dtype=np.float32)[:, None] / horizon) * self.frequency

        self.backcast_cos_template  = K.transpose(K.cos(self.backcast_grid))

        self.backcast_sin_template  = K.transpose(K.sin(self.backcast_grid))
        self.forecast_cos_template  = K.transpose(K.cos(self.forecast_grid))
        self.forecast_sin_template  = K.transpose(K.sin(self.forecast_grid))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        x = self.theta(x)
        params_per_harmonic    = self.theta_size // 4
        backcast_harmonics_cos = K.dot(inputs[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = K.dot(x[:, 3 * params_per_harmonic:], 
                                       self.backcast_sin_template)
        backcast               = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = K.dot(x[:, :params_per_harmonic], 
                                       self.forecast_cos_template)
        forecast_harmonics_sin = K.dot(x[:, params_per_harmonic:2 * params_per_harmonic], 
                                       self.forecast_sin_template)
        forecast               = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast