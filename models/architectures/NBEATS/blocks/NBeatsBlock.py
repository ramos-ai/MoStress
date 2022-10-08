from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from models.architectures.NBeats.blocks.GenericBlock import GenericBlock
from models.architectures.NBeats.blocks.TrendBlock import TrendBlock
from models.architectures.NBeats.blocks.SeasonalBlock import SeasonalBlock

class NBeatsBlock(keras.layers.Layer):

    def __init__(self,
                 model_type           = 'generic',
                 lookback             = 7,
                 horizon              = 1,
                 num_generic_neurons  = 512,
                 num_generic_stacks   = 30,
                 num_generic_layers   = 4,
                 num_trend_neurons    = 256,
                 num_trend_stacks     = 3,
                 num_trend_layers     = 4,
                 num_seasonal_neurons = 2048,
                 num_seasonal_stacks  = 3,
                 num_seasonal_layers  = 4,
                 num_harmonics        = 1,
                 polynomial_term      = 3,
                 **kwargs):
        super(NBeatsBlock, self).__init__()
        """Final N-Beats model that combines different blocks.  Inputs:
            model_type: str -> type of architecture to use.  Must be one of
                               ['generic', 'interpretable']
            ----
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon: int -> How far out into the future you would like
                             your predictions to be
            ----
            num_generic_neurons: int -> size of dense layers in generic block
            ----
            num_generic_stacks: int -> number of generic blocks to stack on top
                             of one another
            ----
            num_generic_layers: int -> number of dense layers to store inside a
                             generic block
            ----
            num_trend_neurons: int -> size of dense layers in trend block
            ----
            num_trend_stacks: int -> number of trend blocks to stack on top of
                             one another
            ----
            num_trend_layers: int -> number of Dense layers inside a trend block
            ----
            num_seasonal_neurons: int -> size of Dense layer in seasonal block
            ----
            num_seasonal_stacks: int -> number of seasonal blocks to stack on top
                             on top of one another
            ----
            num_seasonal_layers: int -> number of Dense layers inside a seasonal
                             block
            ----
            num_harmonics: int -> seasonal term to use for seasonal stack
            ----
            polynomial_term: int -> size of polynomial expansion for trend block
            """
        self.model_type           = model_type
        self.lookback             = lookback
        self.horizon              = horizon
        self.num_generic_neurons  = num_generic_neurons
        self.num_generic_stacks   = num_generic_stacks
        self.num_generic_layers   = num_generic_layers
        self.num_trend_neurons    = num_trend_neurons
        self.num_trend_stacks     = num_trend_stacks
        self.num_trend_layers     = num_trend_layers
        self.num_seasonal_neurons = num_seasonal_neurons
        self.num_seasonal_stacks  = num_seasonal_stacks
        self.num_seasonal_layers  = num_seasonal_layers
        self.num_harmonics        = num_harmonics
        self.polynomial_term      = polynomial_term
    
        # Model architecture is pretty simple: depending on model type, stack
        # appropriate number of blocks on top of one another
        # default values set from page 26, Table 18 from paper
        if model_type == 'generic':
            self.blocks_ = [GenericBlock(lookback       = lookback, 
                                         horizon        = horizon,
                                         num_neurons    = num_generic_neurons, 
                                         block_layers   = num_generic_layers)
                             for _ in range(num_generic_stacks)]
        if model_type == 'interpretable':
            self.blocks_ = [TrendBlock(lookback         = lookback,
                                       horizon          = horizon,
                                       num_neurons      = num_trend_neurons,
                                       block_layers     = num_trend_layers, 
                                       polynomial_term  = polynomial_term)
                            for _ in range(num_trend_stacks)] + [
                            SeasonalBlock(lookback      = lookback,
                                          horizon       = horizon,
                                          num_neurons   = num_seasonal_neurons,
                                          block_layers  = num_seasonal_layers,
                                          num_harmonics = num_harmonics)
                            for _ in range(num_seasonal_stacks)]
        
    def call(self, inputs):
        residuals = K.reverse(inputs, axes = 0)
        forecast  = inputs[:, -1:]
        residual_collection, i = {}, 0
        for block in self.blocks_:
            backcast, block_forecast = block(residuals)
            residuals = keras.layers.Subtract()([residuals, backcast])
            forecast  = keras.layers.Add()([forecast, block_forecast])
            residual_collection[i], i = residuals, i + 1
        return forecast, residual_collection