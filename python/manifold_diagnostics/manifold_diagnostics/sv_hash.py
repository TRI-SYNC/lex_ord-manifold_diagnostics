import numpy as np
from numpy.linalg import svd
import hashlib

def top_singular_vectors_hash(A, k=3, decimals=6, solver='svd', seed=0):
    A = np.asarray(A, dtype=float)
    U, S, Vt = svd(A, full_matrices=False)
    V = Vt[:k].T  # shape: n_features x k
    for i in range(V.shape[1]):
        col = V[:, i]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            V[:, i] = -col
    Q = np.round(V, decimals=decimals)
    meta = f"k={k};decimals={decimals};solver={solver};seed={seed};shape={Q.shape}"
    payload = meta.encode('utf-8') + Q.tobytes()
    h = hashlib.sha256(payload).hexdigest()
    return h, meta
