import numpy as np
from manifold_diagnostics.projection import projection_distance_linear

def test_projection_linear():
    U = np.array([[1.0, 0.0],[0.0, 1.0]])
    x = np.array([1.0, 2.0])
    d, p = projection_distance_linear(x, U)
    assert abs(d) < 1e-12
    assert np.allclose(p, x)
