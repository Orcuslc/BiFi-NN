import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from nn import Reduced_NN
from utils import compute_reduced_solution

class PODNN(Reduced_NN):
    """
        The class of PODNN model
    """
    def __init__(self, *, reduced_dimension, param_dimension, configurations):
        """init
        
        Arguments:
            reduced_dimension {int} -- The dimension of reduced space
            param_dimension {int} -- The dimension of parameter space
            configurations {dict or list(dict)} -- The configuration of neural networks, either a dictionary (in which case the configurations for each neural networks will be the same), or a list of dictionarys, each contain a configurations for the corresponding network
        """
        if type(configurations) is dict:
            configurations = [configurations]*reduced_dimension
        assert(len(configurations) == reduced_dimension, "`configurations` should either be a single dict or a list of dicts and the length equal to `reduced_dimension`")
        for i in range(reduced_dimension):
            configurations["layers"].insert(param_dimension)
            configurations["layers"].append(i+1)
        super().__init__(reduced_dimension, configurations)

    def train(self, param, hifi_coefficients, L = None, **kwargs):
        """Training the neural networks
        
        Arguments:
            param {np.ndarray} -- Training input: parameter
            hifi_coefficients {np.ndarray} -- Hi-fidelity POD coefficients, of shape [N_sample, N_dimension]
        
        Keyword Arguments:
            L {int or iterable} -- The sequence of models to train (default: {None} -- train all models)
        """
        if L is None:
            L = range(self._dimension)
        if type(L) is int:
            L = [L]
        param = [param]*len(L)
        x = [param]*len(L)
        y = [hifi_coefficients[:, :(i+1)] for i in L]
        for xi, yi, index in zip(x, y, L):
            self._models[index].fit(xi, yi, **kwargs)

    def predict(self, param, Lmax = self._dimension, **kwargs):
        x = [param]*Lmax
        return super().predict(x, Lmax, **kwargs)

    def approximate_reduced_solution(self, param, basis):
        L = basis.shape[0]
        coefficients = self.predict(param, L)
        reduced_solution = []
        for i in range(L):
            reduced_solution.append(compute_reduced_solution(self.coefficients[i], basis[:i, :]))
        return reduced_solution

    def evaluate(self, )