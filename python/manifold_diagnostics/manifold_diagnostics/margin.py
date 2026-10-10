import numpy as np

def linear_margin(x, w, b):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    b = float(b)
    raw = float(np.dot(w, x) + b)
    denom = np.linalg.norm(w)
    if denom == 0:
        return raw, None
    return raw / denom, raw
