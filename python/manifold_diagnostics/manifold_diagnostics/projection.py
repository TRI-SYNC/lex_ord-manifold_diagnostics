import numpy as np

def project_to_subspace(x, U):
    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)
    return U @ (U.T @ x)

def projection_distance_linear(x, U):
    proj = project_to_subspace(x, U)
    return float(np.linalg.norm(x - proj)), proj

def projection_distance_nonlinear(x, project_fn, max_iters=50, tol=1e-6):
    x = np.asarray(x, dtype=float)
    prev = x.copy()
    converged = False
    for i in range(max_iters):
        p = project_fn(prev)
        p = np.asarray(p, dtype=float)
        dist = np.linalg.norm(x - p)
        if np.linalg.norm(prev - p) < tol:
            converged = True
            break
        prev = p
    return float(np.linalg.norm(x - p)), bool(converged), int(i+1), p
