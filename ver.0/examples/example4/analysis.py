import sys
sys.path.append("../")

from handle_results import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("H", type = int)
parser.add_argument("N", type = int)
args = parser.parse_args()

H = args.H
N = args.N

podnn_path = "results/podnn/N_{0}_H_{1}.npy".format(N, H)
mpodnn_path = "results/mpodnn/N_{0}_H_{1}.npy".format(N, H)
bifinn_path = "results/bifinn/N_{0}_H_{1}.npy".format(N, H)
mbifinn_path = "results/mbifinn/N_{0}_H_{1}.npy".format(N, H)
podnn_res = read_results(podnn_path)
mpodnn_res = read_results(mpodnn_path)
bifinn_res = read_results(bifinn_path)
mbifinn_res = read_results(mbifinn_path)

import matplotlib.pyplot as plt
plt.semilogy(podnn_res["approx_errors"], label = "podnn")
plt.semilogy(mpodnn_res["approx_errors"], label = "mpodnn")
plt.semilogy(bifinn_res["approx_errors"], label = "bifinn")
plt.semilogy(mbifinn_res["approx_errors"], label = "mbifinn")
plt.legend()
plt.xlabel("Number of POD basis")
plt.ylabel("Rel. Approx. Error")
plt.title("N={0},H={1}".format(N, H))
plt.show()
