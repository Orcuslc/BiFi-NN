import sys
sys.path.append("../")

from handle_results import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-N", type = int)
parser.add_argument("-c", "--criterion", type = str, choices = ["last", "best"])
args = parser.parse_args()
N = args.N

Hs = [8, 16, 24, 32]
podnn = []
mpodnn = []
bifinn = []
mbifinn = []
for H in Hs:
    podnn_path = "results/podnn/N_{0}_H_{1}.npy".format(N, H)
    mpodnn_path = "results/mpodnn/N_{0}_H_{1}.npy".format(N, H)
    bifinn_path = "results/bifinn/N_{0}_H_{1}.npy".format(N, H)
    mbifinn_path = "results/mbifinn/N_{0}_H_{1}.npy".format(N, H)
    podnn_res = read_results(podnn_path)
    mpodnn_res = read_results(mpodnn_path)
    bifinn_res = read_results(bifinn_path)
    mbifinn_res = read_results(mbifinn_path)
    podnn.append(podnn_res["approx_errors"])
    mpodnn.append(mpodnn_res["approx_errors"])
    bifinn.append(bifinn_res["approx_errors"])
    mbifinn.append(mbifinn_res["approx_errors"])

import numpy as np
podnn = np.asarray(podnn)
mpodnn = np.asarray(mpodnn)
bifinn = np.asarray(bifinn)
mbifinn = np.asarray(mbifinn)

if args.criterion == "last":
    podnn_index = np.argmin(podnn[:, -1])
    mpodnn_index = np.argmin(mpodnn[:, -1])
    bifinn_index = np.argmin(bifinn[:, -1])
    mbifinn_index = np.argmin(mbifinn[:, -1])
    podnn = podnn[podnn_index, :]
    mpodnn = mpodnn[mpodnn_index, :]
    bifinn = bifinn[bifinn_index, :]
    mbifinn = mbifinn[mbifinn_index, :]
else:
    podnn_index = np.argmin(podnn, axis = 0)
    mpodnn_index = np.argmin(mpodnn, axis = 0)
    bifinn_index = np.argmin(bifinn, axis = 0)
    mbifinn_index = np.argmin(mbifinn, axis = 0)
    podnn = np.min(podnn, axis = 0)
    mpodnn = np.min(mpodnn, axis = 0)
    bifinn = np.min(bifinn, axis = 0)
    mbifinn = np.min(mbifinn, axis = 0)

import matplotlib.pyplot as plt
if args.criterion == "last":
    plt.semilogy(podnn, label = "podnn,H={0}".format(Hs[podnn_index]))
    plt.semilogy(mpodnn, label = "mpodnn,H={0}".format(Hs[mpodnn_index]))
    plt.semilogy(bifinn, label = "bifinn,H={0}".format(Hs[bifinn_index]))
    plt.semilogy(mbifinn, label = "mbifinn,H={0}".format(Hs[mbifinn_index]))
else:
    plt.semilogy(podnn, label = "podnn")
    plt.semilogy(mpodnn, label = "mpodnn")
    plt.semilogy(bifinn, label = "bifinn")
    plt.semilogy(mbifinn, label = "mbifinn")
plt.legend()
plt.xlabel("Number of POD basis")
plt.ylabel("Rel. Approx. Error")
plt.title("N={0}, criterion={1}".format(N, args.criterion))
plt.show()