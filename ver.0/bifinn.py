from podnn import PODNN, Modified_PODNN
from nn import NN, _build_logger, log
import os
from utils import compute_error
from pod import compute_reduced_solution
from functools import wraps, partial
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import copy

class BiFiNN(PODNN):
	def __init__(self, z_shape, Lmax, layers, n_start = 10, save_path = "models/bifinn", status = "train", logfile = "bifinn.log"):
		z_shape += Lmax
		super().__init__(z_shape, Lmax, layers, n_start, save_path, status, logfile)

	def train_and_store(self, train_data, *, batch_size, epochs, **kwargs):
		train_data = self.extend_data(train_data)
		return super().train_and_store(train_data, batch_size = batch_size, epochs = epochs, **kwargs)
	
	def train(self, train_data, *, batch_size, epochs, **kwargs):
		train_data = self.extend_data(train_data)
		super().train(train_data, batch_size = batch_size, epochs = epochs, **kwargs)
	
	def predict(self, predict_data):
		predict_data = self.extend_data(predict_data)
		return super().predict(predict_data)

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

	def load_and_predict(self, predict_data, path=None):
		predict_data = self.extend_data(predict_data)
		return super().load_and_predict(predict_data, path=path)
	
	@log
	def load_and_test(self, test_data, path=None):
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

	def extend_data(self, data):
		new_data = copy.deepcopy(data)
		new_data["z"] = np.concatenate([data["z"], data["c_low"]], axis = 1)
		return new_data


class Modified_BiFiNN(Modified_PODNN):
	def __init__(self, z_shape, Lmax, layers, n_start = 10, save_path = "models/mbifinn", status = "train", logfile = "mbifinn.log"):
		z_shape += Lmax
		super().__init__(z_shape, Lmax, layers, n_start, save_path, status, logfile)

	def train_and_store(self, train_data, *, batch_size, epochs, **kwargs):
		train_data = self.extend_data(train_data)
		return super().train_and_store(train_data, batch_size = batch_size, epochs = epochs, **kwargs)

	def train(self, train_data, *, batch_size, epochs, **kwargs):
		train_data = self.extend_data(train_data)
		super().train(train_data, batch_size = batch_size, epochs = epochs, **kwargs)

	def predict(self, predict_data):
		predict_data = self.extend_data(predict_data)
		return super().predict(predict_data)

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

	def load_and_predict(self, predict_data, path=None):
		predict_data = self.extend_data(predict_data)
		return super().load_and_predict(predict_data, path=path)

	def load_and_test(self, test_data, path=None):
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

	def extend_data(self, data):
		new_data = copy.deepcopy(data)
		new_data["z"] = np.concatenate([data["z"], data["c_low"]], axis = 1)
		return new_data


if __name__ == "__main__":
	from preprocessing import prepare_data
	bifinn = Modified_BiFiNN(z_shape = 10, 
				Lmax = 2, 
				layers = [[16, 16] for i in range(2)], 
				n_start = 1)
	train_data, test_data = prepare_data("examples/example4/dataset.mat", 
				L = 2, 
				train_index = range(500), 
				test_index = range(500, 600), 
				basis_index = range(600, 880))
	bifinn.train(train_data, 
				batch_size=100, 
				epochs=10, 
				verbose = 0)
	print(bifinn.load_and_test(test_data))