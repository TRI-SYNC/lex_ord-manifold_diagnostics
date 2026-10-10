import numpy as np
from manifold_diagnostics.jacobian import finite_diff_jacobian, jacobian_sigma_max

def f(x):
    A = np.array([[2.0, 0.0],[0.0, 3.0]])
    return A @ x

def test_finite_diff_jacobian():
    x = np.array([1.0, 2.0])
    J = finite_diff_jacobian(f, x)
    assert np.allclose(J, np.array([[2.0, 0.0],[0.0, 3.0]]), atol=1e-6)

def test_jacobian_sigma_max():
    J = np.array([[2.0, 0.0],[0.0, 3.0]])
    sigma = jacobian_sigma_max(J, iters=200, tol=1e-9, seed=1)
    assert abs(sigma - 3.0) < 1e-6
