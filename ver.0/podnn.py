from nn import NN, _build_logger, log
import os
from utils import compute_error
from pod import compute_reduced_solution
from functools import wraps, partial

class PODNN:
    def __init__(self, z_shape, Lmax, layers, n_start = 10, save_path = "models/podnn", logfile = "podnn.log"):
        assert Lmax == len(layers), "Length of layers must equal to Lmax"
        self.Lmax = Lmax
        self.n_start = n_start
        self.save_path = save_path
        self._logger = _build_logger(logfile)
        os.makedirs(self.save_path, exist_ok = True)
        self._build_models(z_shape, layers)

    @log 
    def _build_models(self, z_shape, layers):
        self.nns = []
        for i in range(self.Lmax):
            self.nns.append(NN([z_shape] + layers[i] + [i+1], multi_start = self.n_start))
    
    @log
    def train(self, train_data, *, batch_size, epochs, **kwargs):
        for i in range(self.Lmax):
            print("Training NN for Basis {0}".format(i+1))
            self.nns[i].train(x = train_data["z"], 
                            y = train_data["c_high"][:, :(i+1)],
                            batch_size = batch_size,
                            epochs = epochs,
                            **kwargs)
            self.nns[i].save("{0}/basis_{1}".format(self.save_path, i))
        print("Training finished")

    @log
    def test(self, test_data):
        V_high = test_data["V_high"]
        u_high = test_data["u_high"]
        c_high = test_data["c_high"]
        c_pred = []
        u_pred = []
        coeff_errors = []
        approx_errors = []
        for i in range(self.Lmax):
            c = self.nns[i].best_model.predict(test_data["z"])
            u = compute_reduced_solution(c, V_high[:(i+1), :])
            c_pred.append(c)
            u_pred.append(u)
            coeff_errors.append(compute_error(c_high[:, :(i+1)], c, scale = u_high))
            approx_errors.append(compute_error(u_high, u, scale = u_high))
        return {"c": c_pred,
                "u": u_pred,
                "coeff_errors": coeff_errors,
                "approx_errors": approx_errors}

    @log
    def load(self):
        self.nns = []
        for i in range(self.Lmax):
            self.nns.append(NN([1, 1, 1], multi_start = self.n_start))
            self.nns[-1].load()

if __name__ == "__main__":
    from preprocessing import prepare_data
    podnn = PODNN(z_shape = 10, Lmax = 2, layers = [[16, 16] for i in range(2)], n_start = 1)
    train_data, test_data = prepare_data("examples/example4/dataset.mat", L = 2, train_index = range(500), test_index = range(500, 600), basis_index = range(600, 880))
    podnn.train(train_data, batch_size = 100, epochs = 10, verbose = 0)
    print(podnn.test(test_data))