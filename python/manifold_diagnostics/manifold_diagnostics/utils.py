import numpy as np

def finite_diff_eps(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
