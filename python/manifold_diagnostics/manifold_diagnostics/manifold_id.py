import numpy as np

def identify_manifold(x, manifolds, beta=10.0, ambiguous_rel_tol=0.05):
    dists = []
    for m in manifolds:
        d = float(m['dist_fn'](x))
        dists.append((m['id'], d))
    dists.sort(key=lambda t: t[1])
    best_id, best_d = dists[0]
    second_d = dists[1][1] if len(dists) > 1 else np.inf
    ambiguous = (second_d - best_d) / max(1e-12, abs(best_d)) < ambiguous_rel_tol
    vals = np.array([-beta * d for _, d in dists])
    exp = np.exp(vals - np.max(vals))
    probs = exp / exp.sum()
    conf = float(probs[0])
    return {'manifold_id': best_id, 'distance': best_d, 'confidence': conf, 'ambiguous': bool(ambiguous)}
