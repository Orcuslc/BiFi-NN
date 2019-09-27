import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from ../utils import compute_error

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
            configurations {dict} -- Configurations of the networks, see `Reduced_NN._build_networks` for details
        """
        self._dimension = dimension
        self._build_models(configurations)

    def _build_models(self, configurations):
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
        assert(len(configurations) == self._dimension or type(configurations) == dict, "`configurations` should either be a single dictionary or a list of dictionaries whose length equals to `self._dimension")
        if type(configurations) == dict:
            configurations = [configurations] * self._dimension
        self._models = []
        for i in range(self._dimension):
            layers = configurations[i]["layers"]
            activation = configurations[i].get("activation", "tanh")
            optimizer = configurations[i].get("optimizer", "Adam")
            self._models.append(build_dense_network(layers, activation, optimizer))
    
    def train(self, x, y, L = None, **kwargs):
        """Training models
        
        Arguments:
            x {list(np.ndarray)} -- A list of inputs for each network
            y {list(np.ndarray)} -- A list of outputs for each network
            
        Keyword Arguments:
            L {int or iterable} -- Sequence numbers of models to be trained (default: {None} -- ALL models)

        Raises:
            AssertionError -- Length of `L` should be the same with length of `x` and `y`

        Notes:
            Overloaded by each subclass
        """
        if L is None:
            L = range(self._dimension)
        if type(L) is int:
            L = [L]
        if type(x) is np.ndarray:
            x = [x]
        if type(y) is np.ndarray:
            y = [y]

        assert(len(L) == len(x) and len(L) == len(y), "The length of `L` should be the same with length of `x` and `y`")
        for xi, yi, index in zip(x, y, L):
            self._models[index].fit(xi, yi, **kwargs)
 
    # def _predict_one_model(self, x, *, L = -1, **kwargs):
    #     """Predict for the L^th model
        
    #     Arguments:
    #         x {np.array} -- Input for the L^th network
        
    #     Keyword Arguments:
    #         L {int} -- The number of model to predict (default: {None: -1})
        
    #     Returns:
    #         Y {np.array} -- The predicted output of the L^th network
    #     """
    #     return self._models[L].predict(x, **kwargs)

    def predict(self, x, Lmax = self._dimension, **kwargs):
        """Predict for the whole model
        
        Arguments:
            x {list(np.ndarray)} -- List of inputs for the models
        
        Keyword Arguments:
            Lmax {int} -- The maximum number of models to predict (default: self._dimension)

        Returns:
            y {list(np.ndarray)} -- list of corresponding outputs

        Raises:
            AssertionError -- The length of `x` should be `L`

        Notes:
            Overloaded by subclasses
        """
        if Lmax == 1:
            x = [x]
        assert(len(x) == Lmax, "The length of `x` should be `L`")
        y = []
        for i in range(Lmax):
            y.append(self._models[i].predict(x[i], **kwargs))
        return y

    def evaluate(self, x, y, L = self._dimension, scale = None, **kwargs):
        """Evaluate the models
        
        Arguments:
            x {list(np.ndarray)} -- The list of inputs for each model
            y {list(np.ndarray)} -- The list of true outputs for each model
        
        Keyword Arguments:
            L {int} -- The maximum number of models to evaluate (default: {self._dimension})
            scale {int or np.ndarray} -- The relative scale to compute the errors (default: {None})
        
        Returns:
            errors {list(np.ndarray)} -- The list of errors for each model

        Raises:
            AssertionError -- The length of `x` and `y` should be `L`
        """
        if L == 1:
            y = [y]
        assert(len(x) == L and len(y) == L, "The length of `x` and `y` should be `L`")
        y_pred = self.predict(x, L, **kwargs)
        errors = []
        for i in range(L):
            errors.append(compute_error(y_pred[i], y[i], scale = scale))
        return errors

    def save(self, path):
        """Save the models
        
        Arguments:
            path {str} -- The path to the folder
        """
        for i in range(self._dimension):
            self._models[i].save(path + "/{0}_model_{1}.h5".format(self.method, i))

    def load(self, path):
        for i in range(self._dimension):
            self._models[i] = keras.models.load_model(path + "/{0}_model_{1}.h5".format(self.method, i))
    
    @classmethod
    @property
    def method(self):
        return self.__name__

    @property
    def models(self):
        return self._models
    

        
        