import numpy as np
from utils import *
from pod import *

def prepare_data(path, L, train_index, test_index, basis_index):
    u_high, u_low, z = read_mat(path)
    u_high_train = u_high[train_index, :]
    u_high_test = u_high[test_index, :]
    u_low_train = u_low[train_index, :]
    u_low_test = u_low[test_index, :]
    z_train = z[train_index, :]
    z_test = z[test_index, :]
    V_high = pod(u_high[basis_index, :], L)
    V_low = pod(u_low[basis_index, :], L)
    c_high_train = compute_reduced_coefficient(u_high_train, V_high)
    c_high_test = compute_reduced_coefficient(u_high_test, V_high)
    c_low_train = compute_reduced_coefficient(u_low_train, V_low)
    c_low_test = compute_reduced_coefficient(u_low_test, V_low)    
    train_data = {"u_high": u_high_train,
                "u_low": u_low_train,
                "c_high": c_high_train,
                "c_low": c_low_train,
                "z": z_train,
                "V_high": V_high, 
                "V_low": V_low
    }
    test_data = {"u_high": u_high_test,
                "u_low": u_low_test,
                "c_high": c_high_test,
                "c_low": c_low_test,
                "z": z_test,
                "V_high": V_high,
                "V_low": V_low
    }
    return train_data, test_data

if __name__ == "__main__":
    path = "examples/example4/dataset.mat"
    L = 16
    train_index = range(500)
    test_index = range(500, 600)
    basis_index = range(600, 880)
    train_data, test_data = prepare_data(path, L, train_index, test_index, basis_index)