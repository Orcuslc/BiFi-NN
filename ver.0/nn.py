import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.losses import MSE
import tensorflow.keras.backend as K
from functools import wraps
import numpy as np
import os
import logging
import time
# from scipy.optimize import fmin_l_bfgs_b

class NN:
	"""
	Base class for PODNN and BiFiNN, with input and output fixed
	"""
	def __init__(self, layers, multi_start = 10):
		self.multi_start = multi_start
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
			hist = model.fit(x, y, 
			batch_size, 
			epochs, 
			callbacks = [keras.callbacks.EarlyStopping(patience = 500, 
										restore_best_weights = True),
						keras.callbacks.tensorboard_v1.Tensorboard(log_dir = "./tensorboard", 
										histogram_freq = 50,
										write_grads = True)],
			validation_split = 0.2, **kwargs)
			val_losses.append(min(hist.history["val_loss"]))
		self.best_index = val_losses.index(min(val_losses))
		self.best_model = self.models[self.best_index]

	def save(self, path):
		os.makedirs(path, exist_ok = True)
		for (i, model) in zip(range(self.multi_start), self.models):
			model.save_weights("{0}/weights_{1}.h5".format(path, i))
			model_json = model.to_json()
			with open("{0}/model_{1}.json".format(path, i), "w") as f:
				f.write(model_json)
		self.best_model.save_weights("{0}/best_weights_{1}.h5".format(path, self.best_index))
		best_model_json = self.best_model.to_json()
		with open("{0}/best_model_{1}.json".format(path, self.best_index), "w") as f:
			f.write(best_model_json)

	def load(self, path):
		self.models = []
		for i in range(self.multi_start):
			with open("{0}/model_{1}.json".format(path, i), "r") as f:
				model_json = f.read()
			model = model_from_json(model_json)
			model.load_weights("{0}/weights_{1}.h5".format(path, i))
			self.models.append(model)
		self.load_best(path)

	def load_best(self, path):
		files = os.listdir(path)
		for file in files:
			if file.startswith("best_model"):
				with open("{0}/{1}".format(path, file), "r") as f:
					model_json = f.read()
				self.best_model = model_from_json(model_json)
		for file in files:
			if file.startswith("best_weights"):
				self.best_model.load_weights("{0}/{1}".format(path, file))

def _build_logger(logfile):
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler(logfile)
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(asctime)s]-[%(name)s]-[%(levelname)s]  %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger

def log(func):
	@wraps(func)
	def wrapper(self, *args, **kwargs):
		self._logger.info("start executing: {0}".format(func.__name__))
		start = time.time()
		res = func(self, *args, **kwargs)
		end = time.time()
		self._logger.info("finished executing: {0}, time span: {1}".format(func.__name__, end - start))
		return res
	return wrapper

# class Logger:
# 	def __init__(self, logfile):
# 		self._build_logger(logfile)		

# 	def _build_logger(self, logfile):
# 		logger = logging.getLogger(__name__)
# 		logger.setLevel(logging.INFO)
# 		handler = logging.FileHandler(logfile)
# 		handler.setLevel(logging.INFO)
# 		formatter = logging.Formatter('[%(asctime)s]-[%(name)s]-[%(levelname)s]  %(message)s')
# 		handler.setFormatter(formatter)
# 		logger.addHandler(handler)
# 		self._logger = logger

# 	def log(self):
# 		def decorate(func):
# 			@wraps(func)
# 			def wrapper(*args, **kwargs):
# 				self._logger.info("start executing: {0}".format(func.__name__))
# 				start = time.time()
# 				res = func(*args, **kwargs)
# 				end = time.time()
# 				self._logger.info("finished executing: {0}, time span: {1}".format(func.__name__, end - start))
# 				return res
# 			return wrapper
# 		return decorate


	# def loss(c_true, c_pred):
	# 	self.loss = MSE(c_true, c_pred)
	# 	return self.loss

	# def grad(self):
	# 	return K.gradients(self.loss, )
