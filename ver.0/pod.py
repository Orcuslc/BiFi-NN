import numpy as np

def pod(data, L = None, epsilon = None, scale = 2):
    """Proper Orthogonal Decomposition for a snapshot matrix
    
    Arguments:
        data {np.ndarray} -- The snapshot matrix of shape [N_sample, N_feature], where each row is a sample point
    
    Keyword Arguments:
        L {int} -- The dimension of reduced space (default: {None})
        epsilon {float} -- The residue of singular values (default: {None})
        scale {1 or 2} -- The exponential scale used to compute the residue (default: {2})

    Returns:
        V {np.ndarray} -- The POD basis of shape [L, N_feature], where each row is a POD basis
    
    Raises:
        AssertionError -- `L` and `epsilon` cannot be None simultaneously
    """
    assert L is not None or epsilon is not None, "L and epsilon cannot be None simultaneously"
    _, S, V = np.linalg.svd(data, full_matrices=False)
    if L is None:
        S = np.cumsum(S**scale)
        L = np.where(S/S[-1] >= 1-epsilon)[0][0]
    return V[:L, :]

def compute_reduced_coefficient(u, V):
    """Project a full-order solution onto the POD basis to get the POD coefficients
    
    Arguments:
        u {np.ndarray} -- Full-order solution, of shape [N_sample, N_feature]
        V {np.ndarray} -- POD basis, of shape [N_basis, N_feature]
    
    Returns:
        c -- POD coefficients, of shape [N_sample, N_basis]
    """
    return u.dot(V.transpose())

def compute_reduced_solution(c, V):
    """Compute the reduced solution by a linear combination of POD basis
    
    Arguments:
        c {np.ndarray} -- POD coefficients, of shape [N_sample, N_basis]
        V {np.ndarray} -- POD basis, of shape [N_basis, N_feature]
    
    Returns:
        u -- Reduced solution, of shape [N_sample, N_feature]
    """
    return c.dot(V)