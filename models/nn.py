import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np

def build_dense_network(layers, activation = K.tanh, optimizer = keras.optimizers.Adam()):
    """Build a fully connected neural network
    
    Arguments:
        layers {list} -- Number of neurons in each layer, including the input and output layers
    
    Keyword Arguments:
        activation {function or string} -- Activation function (default: {K.tanh})
        optimizer {object(keras.optimizers.Optimizer) or string} -- Optimizer (default: {keras.optimizers.Adam()})
    
    Returns:
        model {object(keras.models.Model)} -- The compiled neural network
    """
    model = keras.models.Sequential([
        keras.layers.Dense(layers[1], input_shape = (layers[0], ), activation = activation)
    ])
    for i in range(2, len(layers)-2):
        model.add(keras.layers.Dense(layers[i], activation = activation))
    model.add(keras.layers.Dense(layers[-1], activation = "linear"))
    model.compile(loss = "mse",
                optimizer = optimizer)
    return model

class Reduced_NN:
    """
    Base class for reduced-basis neural networks
    """
    def __init__(self, dimension, configurations):
        """init
        
        Arguments:
            dimension {int} -- Dimension of reduced space
            configurations {dict} -- Configurations of the networks, see `_build_networks` for details
        """
        self._dimension = dimension
        self._build_networks(configurations)

    def _build_networks(self, configurations):
        """Build networks
        
        Arguments:
            configurations {list(dict)} -- Configurations of the network, each dict should contain the following keys:
                layers {list} -- the shape of dense layers for each network, including the input and output layer
                activation {function or string} -- the activation function used in each dense layer, default {"tanh"}
                optimizer {keras.optimizers.Optimizer or string} -- the optimizer used for the nn model, default {"Adam"}
            if len(configurations) = 1, then the default setting is that all networks share the same configurations

        Return:
            self._models {list(keras.models.Model)} -- the models for each reduced network
        """
        assert(len(configurations) == self._dimension or len(configurations) == 1, "The length of `configurations` should either be 1 or equal to the dimension")
        if len(configurations) == 1:
            configurations = configurations * self._dimension
        self._models = []
        for i in range(self._dimension):
            layer = configurations[i]["layers"]
            activation = configurations[i].get("activation", "tanh")
            optimizer = configurations[i].get("optimizer", "Adam")
            self._models.append(build_dense_network(layers, activation, optimizer))
    
    def train():
        pass

    def predict():
        pass

    def save():
        pass

    def load():
        pass
    
        
        