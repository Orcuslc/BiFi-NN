import numpy as np
import scipy.io as sio

def read_mat(path):
    """Read MAT files
    
    Arguments:
        path {string} -- The path to the MAT file
    
    Returns:
        u_high, u_low, z {np.ndarray} -- high-fidelity solution, low-fidelity solution, parameter; of the shape 
        [N_sample, N_feature], that is, each row is a sample point

    Notes:
        In MATLAB, the files are stored in a way such that each column is a sample point. So we need to transpose the sample matrix to make it consistent with Python formats.
    """
    data = sio.loadmat(path)
    u_high = data["u_high"].transpose()
    u_low = data["u_low"].transpose()
    z = data["z"].transpose()
    return u_high, u_low, z

def compute_error(true, pred, scale = None):
    """Compute error between the true vector and the predicted vector
    
    Arguments:
        true {np.ndarray} -- The true vector
        pred {np.ndarray} -- The predicted vector
    
    Keyword Arguments:
        scale {np.ndarray} -- (optional) used for computing the relative error (default: {None})
    
    Returns:
        error -- np.ndarray, the error for each prediction (absolute or relative)
    """
    squared_two_norm = lambda x: np.sqrt(np.sum(x**2, axis = 1, keepdims = True))
    error = squared_two_norm(true - pred)
    if scale is not None:
        error /= squared_two_norm(scale)
    return error