from tensorflow.python import keras


class GenericBlock(keras.layers.Layer):
    def __init__(self, lookback=7, horizon=1, num_neurons=512, block_layers=4):
        super(GenericBlock, self).__init__()
        """Generic Block for Nbeats model.  Inputs:  
            lookback: int -> Multiplier you use for horizon to determine
                             how big your training window is
            ----
            horizon:  int -> How far out into the future you would like
                             your predictions to be
            ----
            num_neurons: int -> How many layers to put into each Dense layer in
                                the generic block
            ----
            block_layers: int -> How many Dense layers to add to the block
        """

        # collection of layers in the block
        self.layers_ = [
            keras.layers.Dense(num_neurons, activation="relu")
            for _ in range(block_layers)
        ]
        self.lookback = lookback
        self.horizon = horizon

        # multiply lookback * forecast to get training window size
        self.backcast_size = horizon * lookback

        # numer of neurons to use for theta layer -- this layer
        # provides values to use for backcast + forecast in subsequent layers
        self.theta_size = self.backcast_size + lookback

        # layer to connect to Dense layers at the end of the generic block
        self.theta = keras.layers.Dense(
            self.theta_size, use_bias=False, activation=None
        )

    def call(self, inputs):
        # save the inputs
        x = inputs
        # connect each Dense layer to itself
        for layer in self.layers_:
            x = layer(x)
        # connect to Theta layer
        x = self.theta(x)
        # return backcast + forecast without any modifications
        return x[:, : self.backcast_size], x[:, -self.horizon :]
