import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.losses import MSE
import tensorflow.keras.backend as K
import numpy as np
import os
# from scipy.optimize import fmin_l_bfgs_b

class NN:
	"""
	Base class for PODNN and BiFiNN, with input and output fixed
	"""
	def __init__(self, layers, multi_start = 10, logfile = "log.txt"):
		self.multi_start = multi_start
                self.log = logfile
		self.models = []
		for _ in range(self.multi_start):
			self.models.append(self._build_model(layers))

	def _build_model(self, layers):
		model = Sequential()
		for i in range(len(layers)-1):
			model.add(Dense(layers[i+1], input_dim = layers[i]))
			model.add(Activation('tanh'))
			model.add(BatchNormalization())
		model.add(Dense(layers[-1]))
		opt = keras.optimizers.Adam(lr = 0.01)
		model.compile(optimizer = opt, loss = 'mse')
		return model

	def train(self, x, y, batch_size, epochs, **kwargs):
		val_losses = []
		for model in self.models:
			hist = model.fit(x, y, batch_size, epochs, callbacks = [keras.callbacks.EarlyStopping(patience = 100, restore_best_weights = True)], validation_split = 0.2, **kwargs)
			val_losses.append(min(hist.history["val_loss"]))
		self.best_index = val_losses.index(min(val_losses))
		self.best_model = self.models[self.best_index]
		print("Best model: Model {0}".format(self.best_index))

	def save(self, path):
		os.makedirs(path, exist_ok = True)
		for (i, model) in zip(range(self.multi_start), self.models):
			model.save("{0}/{1}.h5".format(path, i))
		self.best_model.save("{0}/best_{1}.h5".format(path, self.best_index))

	def load(self, path):
		self.models = []
		for i in range(self.multi_start):
			self.models.append(load_model("{0}/{1}.h5".format(path, i)))
		files = os.listdir(path)
		for file in files:
			if file.startswith("best"):
				self.best_model = load_model("{0}/{1}".format(path, file))
				break

	# def loss(c_true, c_pred):
	# 	self.loss = MSE(c_true, c_pred)
	# 	return self.loss

	# def grad(self):
	# 	return K.gradients(self.loss, )
