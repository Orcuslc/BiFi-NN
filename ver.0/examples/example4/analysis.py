import sys
sys.path.append("../")

from handle_results import *

H = 64
N = 400

podnn_path = "results/podnn/N_{0}_H_{1}.npy".format(N, H)
bifinn_path = "results/bifinn/N_{0}_H_{1}.npy".format(N, H)
podnn_res = read_results(podnn_path)
bifinn_res = read_results(bifinn_path)
print(podnn_res["approx_errors"])
print(bifinn_res["approx_errors"])
