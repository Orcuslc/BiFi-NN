import numpy as np
import matplotlib.pyplot as plt

def read_results(path):
    data = np.load(path, allow_pickle=True).tolist()
    c = data["c"]
    u = data["u"]
    coeff_errors = data["coeff_errors"]
    approx_errors = data["approx_errors"]
    return {
        "coeff_errors": [np.mean(x) for x in coeff_errors],
        "approx_errors": [np.mean(x) for x in approx_errors]
    }

def plot_results(results):
    plt.figure(1)
    plt.semilogy(results["approx_errors"])
    plt.show()