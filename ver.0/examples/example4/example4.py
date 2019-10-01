import sys
sys.path.append('../')

import numpy as np
import os
from preprocessing import prepare_data
from podnn import PODNN

H = 16
N = 400

podnn = PODNN(z_shape = 10, Lmax = 16, layers = [[H, H] for i in range(16)], n_start = 10)
train_data, test_data = prepare_data("dataset.mat", L = 2, train_index = range(N*1.25), test_index = range(500, 600), basis_index = range(600, 880))
podnn.train(train_data, batch_size = 100, epochs = 10000, verbose = 1)
res = podnn.test(test_data)
os.makedirs("results/podnn", exist_ok=True)
np.save("results/podnn/{N_{0}_H_{1}.npy".format(N, H), res)
print("ALL Finished")
