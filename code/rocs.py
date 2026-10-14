"""Rank-opposition jumping (Eq. 7) and the rank-opposition crow search ROCS (Algorithm 1)."""
import numpy as np

from model import evaluate


def opposite_rank(busy, v):
    """Eq. (7): the VM whose busy-time rank mirrors the rank of VM v."""
    ranked = np.argsort(busy)
    return int(ranked[len(busy) - 1 - int(np.where(ranked == v)[0][0])])


def rocs(problem, x, rng, budget, J=0.3, AP=0.1, FL=0.5):
    """Algorithm 1 started from seed x with rng = random.Random(seed); equal makespans are accepted."""
    n, m = problem.exe.shape
    makespan, busy = evaluate(problem, x)
    for _ in range(budget):
        i = rng.randrange(n)
        v = x[i]
        if rng.random() < J:
            alt = opposite_rank(busy, v)
        else:
            j = rng.randrange(n)
            if rng.random() >= AP:
                alt = int(v + rng.random() * FL * (x[j] - v))
            else:
                alt = rng.randrange(m)
        if alt == v:
            continue
        x[i] = alt
        candidate, candidate_busy = evaluate(problem, x)
        if candidate <= makespan:
            makespan, busy = candidate, candidate_busy
        else:
            x[i] = v
    return makespan
