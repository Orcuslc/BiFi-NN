from nn import NN, _build_logger, log
import os
from utils import compute_error
from pod import compute_reduced_solution
from functools import wraps, partial
from tensorflow.keras.models import load_model, model_from_json
import tensorflow.keras.backend as K
import time

class PODNN:
	def __init__(self, z_shape, Lmax, layers, n_start = 10, save_path = "models/podnn", status = "train", logfile = "podnn.log"):
		assert Lmax == len(layers), "Length of layers must equal to Lmax"
		self.z_shape = z_shape
		self.Lmax = Lmax
		self.layers = layers
		self.n_start = n_start
		self.save_path = save_path
		self._logger = _build_logger(logfile)
		os.makedirs(self.save_path, exist_ok = True)
		self.best_models = []

	@log 
	def _build_models(self, z_shape, layers):
		self.nns = []
		for i in range(self.Lmax):
			self.nns.append(NN([z_shape] + layers[i] + [i+1], multi_start = self.n_start))

	@log
	def train_and_store(self, train_data, *, batch_size, epochs, **kwargs):
		"""Not Recommended -- VERY SLOW
		"""
		self._build_models(self.z_shape, self.layers)
		for i in range(self.Lmax):
			self._logger.info("Training NN for Basis {0}".format(i+1))
			start = time.time()
			nn = NN([self.z_shape] + self.layers[i] + [i+1], multi_start = self.n_start)
			nn.train(x = train_data["z"], 
							y = train_data["c_high"][:, :(i+1)],
							batch_size = batch_size,
							epochs = epochs,
							**kwargs)
			end = time.time()
			self._logger.info("Finished training, time span: {0}".format(end - start))
			self.best_models.append(self.nns[i].best_model)
			self._logger.info("Saving NN for Basis {0}".format(i+1))
			start = time.time()
			nn.save("{0}/basis_{1}".format(self.save_path, i+1))
			end = time.time()
			self._logger.info("Finished saving, time span: {0}".format(end - start))
			self.nns.append(nn)	

		
	@log
	def train(self, train_data, *, batch_size, epochs, **kwargs):
		for i in range(self.Lmax):
			self._logger.info("Training NN for Basis {0}".format(i+1))
			start = time.time()
			nn = NN([self.z_shape] + self.layers[i] + [i+1], multi_start = self.n_start)
			nn.train(x = train_data["z"], 
							y = train_data["c_high"][:, :(i+1)],
							batch_size = batch_size,
							epochs = epochs,
							**kwargs)
			end = time.time()
			self._logger.info("Finished training, time span: {0}".format(end - start))
			# self.best_models.append(self.nns[i].best_model)
			self._logger.info("Saving NN for Basis {0}".format(i+1))
			start = time.time()
			nn.save("{0}/basis_{1}".format(self.save_path, i+1))
			end = time.time()
			self._logger.info("Finished saving, time span: {0}".format(end - start))
			K.clear_session()

	@log
	def predict(self, predict_data):
		V_high = predict_data["V_high"]
		c_pred = []
		u_pred = []
		for i in range(self.Lmax):
			c = self.best_models[i].predict(predict_data["z"])
			u = compute_reduced_solution(c, V_high[:(i+1), :])
			c_pred.append(c)
			u_pred.append(u)
		return {"c": c_pred,
				"u": u_pred}

	@log
	def test(self, test_data):
		pred = self.predict(test_data)
		c_pred = pred["c"]
		u_pred = pred["u"]
		V_high = test_data["V_high"]
		u_high = test_data["u_high"]
		c_high = test_data["c_high"]
		coeff_errors = []
		approx_errors = []
		for i in range(self.Lmax):
			coeff_errors.append(compute_error(c_high[:, :(i+1)], c_pred[i], scale = u_high))
			approx_errors.append(compute_error(u_high, u_pred[i], scale = u_high))
		return {"c": c_pred,
				"u": u_pred,
				"coeff_errors": coeff_errors,
				"approx_errors": approx_errors}

	@log
	def load_one_all(self, L, path = None):
		nn = NN([1, 1, 1], multi_start = self.n_start)
		if path is None:
			path = self.save_path
		nn.load("{0}/basis_{1}".format(path, L))
		return nn

	@log
	def load_all(self, path = None):
		"""Not Recommended
		"""
		self.nns = []
		self.best_models = []
		for i in range(1, self.Lmax+1):
			self.nns.append(self.load_one_all(i, path))
			self.best_models.append(self.nns[-1].best_model)

	@log
	def load_one_best(self, L, path = None):
		if path is None:
			path = self.save_path
		files = os.listdir("{0}/basis_{1}".format(path, L))
		for file in files:
			if file.startswith("best_model"):
				with open("{0}/basis_{1}/{2}".format(path, L, file), "r") as f:
					model_json = f.read()
				model = model_from_json(model_json)
		for file in files:
			if file.startswith("best_weights"):
				model.load_weights("{0}/basis_{1}/{2}".format(path, L, file))
		return model

	@log
	def load_best(self, path = None):
		"""Still Not Recommended
		"""
		self.best_models = []
		for i in range(1, self.Lmax+1):
			self.best_models.append(self.load_one_best(i, path))

	@log
	def load_and_predict(self, predict_data, path = None):
		"""Recommended for prediction; should be the fastest method to load and test
		"""
		V_high = predict_data["V_high"]
		c_pred = []
		u_pred = []
		for i in range(1, self.Lmax+1):
			model = self.load_one_best(i, path)
			c = model.predict(predict_data["z"])
			K.clear_session() # to speedup loading 
			u = compute_reduced_solution(c, V_high[:i, :])
			c_pred.append(c)
			u_pred.append(u)
		return {"c": c_pred,
				"u": u_pred}

	@log
	def load_and_test(self, test_data, path = None):
		pred = self.load_and_predict(test_data, path)
		c_pred = pred["c"]
		u_pred = pred["u"]
		V_high = test_data["V_high"]
		u_high = test_data["u_high"]
		c_high = test_data["c_high"]
		coeff_errors = []
		approx_errors = []
		for i in range(self.Lmax):
			coeff_errors.append(compute_error(c_high[:, :(i+1)], c_pred[i], scale = u_high))
			approx_errors.append(compute_error(u_high, u_pred[i], scale = u_high))
		return {"c": c_pred,
				"u": u_pred,
				"coeff_errors": coeff_errors,
				"approx_errors": approx_errors}
		

if __name__ == "__main__":
	from preprocessing import prepare_data
	podnn = PODNN(z_shape = 10, Lmax = 2, layers = [[16, 16] for i in range(2)], n_start = 1)
	train_data, test_data = prepare_data("examples/example4/dataset.mat", L = 2, train_index = range(500), test_index = range(500, 600), basis_index = range(600, 880))
	podnn.train(train_data, batch_size = 100, epochs = 10, verbose = 0)
	print(podnn.test(test_data))