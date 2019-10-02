import sys
sys.path.append('../../')
sys.path.append('../')

import numpy as np
import os
from preprocessing import prepare_data
from podnn import PODNN
from bifinn import BiFiNN
from analysis import read_results, plot_results
import matplotlib.pyplot as plt

H = 24
N = 400
Lmax = 16
status = "train"
z_shape = 10
n_start = 10
train_index = range(int(N*1.25))
test_index = range(500, 600)
basis_index = range(600, 880)
epochs = 10000
batch_size = 100
verbose = 1

podnn = PODNN(z_shape = z_shape,
			Lmax = Lmax,
			layers = [[H, H] for i in range(Lmax)],
			n_start = n_start,
			save_path = "models/podnn/N_{0}_H_{1}".format(N, H),
			status = status,
			logfile = "PODNN_N_{0}_H_{1}.log".format(N, H))
bifinn = BiFiNN(z_shape = z_shape,
			Lmax = Lmax,
			layers = [[H, H] for i in range(Lmax)],
			n_start = n_start,
			save_path = "models/bifinn/N_{0}_H_{1}".format(N, H),
			status = status,
			logfile = "BiFiNN_N_{0}_H_{1}.log".format(N, H))
train_data, test_data = prepare_data("dataset.mat",
			L = Lmax,
			train_index = train_index,
			test_index = test_index,
			basis_index = basis_index)
if status == "train":
	podnn.train(train_data,
				batch_size = batch_size,
				epochs = epochs,
				verbose = verbose)
	bifinn.train(train_data,
				batch_size = batch_size,
				epochs = epochs,
				verbose = verbose)

podnn_res = podnn.load_and_test(test_data)
bifinn_res = bifinn.load_and_test(test_data)
os.makedirs("results/podnn", exist_ok = True)
os.makedirs("results/bifinn", exist_ok = True)
np.save("results/podnn/N_{0}_H_{1}.npy".format(N, H), podnn_res)
np.save("results/bifinn/N_{0}_H_{1}.npy".format(N, H), bifinn_res)
print("ALL Finished")
