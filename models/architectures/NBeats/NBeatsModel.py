from tensorflow import keras
from tensorflow.keras import Model
from models.architectures.NBeats.blocks.NBeatsBlock import NBeatsBlock

class NBeatsModel():
    
    def __init__(self, 
                 model_type:str           = 'generic',
                 lookback:int             = 7,
                 horizon:int              = 1,
                 num_generic_neurons:int  = 512,
                 num_generic_stacks:int   = 30,
                 num_generic_layers:int   = 4,
                 num_trend_neurons:int    = 256,
                 num_trend_stacks:int     = 3,
                 num_trend_layers:int     = 4,
                 num_seasonal_neurons:int = 2048,
                 num_seasonal_stacks:int  = 3,
                 num_seasonal_layers:int  = 4,
                 num_harmonics:int        = 1,
                 polynomial_term:int      = 3,
                 loss:str                 = 'mae',
                 learning_rate:float      = 0.001,
                 batch_size: int          = 1024):    
                 
        """
        
        Model used to create and initialize N-Beats model described in the following paper: https://arxiv.org/abs/1905.10437
        
        inputs:
          :model: what model architecture to use.  Must be one of ['generic', 'interpretable']
          :lookback:  what multiplier of the forecast size you want to use for your training window
          :horizon: how many steps into the future you want your model to predict
          :num_generic_neurons: The number of neurons (columns) you want in each Dense layer for the generic block
          :num_generic_stacks: How many generic blocks to connect together
          :num_generic_layers: Within each generic block, how many dense layers do you want each one to have.  If you set this number to 4, and num_generic_neurons to 128, then you will have 4 Dense layers with 128 neurons in each one
          :num_trend_neurons: Number of neurons to place within each Dense layer in each trend block
          :num_trend_stacks: number of trend blocks to stack on top of one another
          :num_trend_layers: number of Dense layers inside a trend block
          :num_seasonal_neurons: size of Dense layer in seasonal block
          :num_seasonal_stacks: number of seasonal blocks to stack on top on top of one another
          :num_seasonal_layers: number of Dense layers inside a seasonal block
          :num_harmonics: seasonal term to use for seasonal stack
          :polynomial_term: size of polynomial expansion for trend block
          :loss: what loss function to use inside keras.  accepts any regression loss function built into keras.  You can find more info here:  https://keras.io/api/losses/regression_losses/
          :learning_rate: learning rate to use when training the model
          :batch_size: batch size to use when training the model
        
        :returns: self
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
        self.loss                 = loss
        self.learning_rate        = learning_rate
        self.batch_size           = batch_size
    
    def get_config(self):
        cfg = super().get_config()
        return cfg
        
    def build_layer(self):
        """
        Initializes the Nested NBeats layer from initial parameters
        
        attributes:
          :model_layer: custom keras layer that contains all of the generic, seasonal and trend layers stacked toger
        
        :returns: self
        
        """
        self.model_layer = NBeatsBlock(**self.__dict__)
        return self
        
    def build_model(self):
        """
        Creates keras model to use for fitting
        
        attributes:
          :model: keras model that contains NBeats model layers as well as inputs, put into the keras Model class
        
        :returns: self
        
        """
        inputs     = keras.layers.Input(shape = (self.horizon * self.lookback, ), dtype = 'float')
        forecasts, residuals = self.model_layer(inputs)
        self.model = Model(inputs, forecasts)
        self.residualModel = Model(inputs, residuals)
        return self
      
    def compile_model(self, run_eagerly=False):
      self.build_layer()
      self.build_model()
      self.model.compile(optimizer = keras.optimizers.Adam(self.learning_rate), 
                          loss      = [self.loss],
                          metrics   = ['mae', 'mape'], run_eagerly=run_eagerly)
      return self
        
    def fit(self, X, y, **kwargs):
        """
        Build and fit model
        
        inputs:
          :X: tensor or numpy array with training windows
          :y: tensor or numpy array with the target values to be predicted
          :kwargs: any additional arguments you'd like to pass to the base keras model
          
        attributes:
          :model_layer: custom keras layer that contains nested Generic, Trend, and Seasonal NBeats blocks
          :model: keras Model class that connects inputs to the model layer
          
        :returns: self  
        """
        self.compile_model()
        self.model.fit(X, y, batch_size = self.batch_size, **kwargs)
        return self
        
    def predict(self, X, **kwargs):
        """
        Passes inputs back to original keras model for prediction
        
        inputs:
          :X: tensor of numpy array with input data
          :kwargs: any additional arguments you'd like to pass to the base keras model
          
        :returns: numpy array that contains model predictions for each sample
        """
        return self.model.predict(X, **kwargs)
    
    
    def evaluate(self, y_true, y_pred, **kwargs):
        """
        Passes predicted and true labels back to the original keras model
        
        inputs:
            :y_true: numpy array or tensorflow with true labels
            :y_pred: numpy array or tensorflow with predicted labels
            :kwargs: any additional arguments you'd like to pass to the base keras model
        
        :returns: list with specified evaluation metrics
        """
        return self.model.evaluate(y_true, y_pred, **kwargs)