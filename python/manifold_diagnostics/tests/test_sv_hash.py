import numpy as np
from manifold_diagnostics.sv_hash import top_singular_vectors_hash

def test_hash_stability():
    rng = np.random.RandomState(0)
    A = rng.randn(10, 5)
    h1, meta1 = top_singular_vectors_hash(A, k=3, decimals=6, seed=0)
    h2, meta2 = top_singular_vectors_hash(A, k=3, decimals=6, seed=0)
    assert h1 == h2
