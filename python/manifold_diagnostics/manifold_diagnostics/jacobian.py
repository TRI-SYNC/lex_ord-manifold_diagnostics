import numpy as np

def finite_diff_jacobian(f, x, eps=None):
    x = np.asarray(x, dtype=float)
    n = x.size
    fx = np.asarray(f(x), dtype=float)
    m = fx.size
    J = np.zeros((m, n), dtype=float)
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
    for i in range(n):
        h = np.zeros_like(x)
        h[i] = eps[i] if np.ndim(eps) else eps
        f_plus = np.asarray(f(x + h), dtype=float)
        f_minus = np.asarray(f(x - h), dtype=float)
        J[:, i] = (f_plus - f_minus) / (2.0 * h[i])
    return J

def jacobian_sigma_max(J, iters=100, tol=1e-6, seed=0):
    J = np.asarray(J, dtype=float)
    m, n = J.shape
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    prev_lambda = 0.0
    for _ in range(iters):
        w = J @ v
        z = J.T @ w
        norm_z = np.linalg.norm(z)
        if norm_z == 0:
            return 0.0
        v = z / norm_z
        lambda_est = v.T @ (J.T @ (J @ v))
        if abs(lambda_est - prev_lambda) / max(1.0, abs(prev_lambda)) < tol:
            break
        prev_lambda = lambda_est
    return float(np.sqrt(max(lambda_est, 0.0)))
