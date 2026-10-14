"""Independent tasks (Section 4.1): datasets, LPT, Min-Min, single-solution CSA, CSA-OBL, CSA-ROJ, and GA.

Task i runs on VM k for length_i / s_k + 300 MB / 1000 MB/s (Eq. 1), and the makespan is the
largest VM completion time. The lower bound of Eq. (2) is total length / total speed plus the mean
transfer time per VM.
"""
import random

import numpy as np

M = 54
SPEEDS = np.array([(400, 500, 600)[k % 3] for k in range(M)], dtype=float)
TRANSFER = 300 / 1000
FL, AP, ITERATIONS = 0.5, 0.1, 20
STRATA = [862, 31, 14, 6, 10, 8, 4, 1, 1, 1, 2, 10, 0, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 0, 0, 0, 1, 3,
          1, 2, 2, 3, 2, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
          1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 1, 1, 0, 0, 0, 0, 1]


def lengths(dataset, n, seed):
    """Task lengths in MI. Simple and stratified follow Farid (2023); Rnaseq uses WfCommons run times."""
    if dataset == "rnaseq":
        from model import generate
        runtime, _ = generate("Rnaseq", n, 3000 + seed)
        return np.asarray(runtime) * 500.0
    r = random.Random(1000 + n + seed)
    if dataset == "simple":
        return np.array([r.randint(10000, 40000) for _ in range(n)], dtype=float)
    per_thousand = n // 1000
    data = []
    low, high = 0, 88000
    for count in STRATA:
        data += [r.randint(low, high) for _ in range(int(per_thousand * count))]
        low, high = high, high + 88000
    return np.array(data, dtype=float)


def run_times(length):
    return length[:, None] / SPEEDS[None, :] + TRANSFER


def lower_bound(length):
    return length.sum() / SPEEDS.sum() + TRANSFER * len(length) / M


def earliest_finish(rt, order):
    """Each task in the given order goes to the VM that finishes it earliest."""
    completion = np.zeros(M)
    x = np.empty(len(order), dtype=int)
    for i in order:
        k = int(np.argmin(completion + rt[i]))
        completion[k] += rt[i, k]
        x[i] = k
    return completion.max(), x


def lpt(rt, length):
    return earliest_finish(rt, np.argsort(-length))


def min_min(rt, length):
    return earliest_finish(rt, np.argsort(length))


def crow_search(rt, x, rng, J=0.0):
    """Single-solution CSA of Farid (2023) with 20 passes over the tasks. With J > 0, a task follows
    Eq. (7) with probability J instead of the crow move (CSA-ROJ, and ROCS when started from LPT)."""
    n = len(x)
    completion = np.zeros(M)
    np.add.at(completion, x, rt[np.arange(n), x])
    makespan = completion.max()
    for _ in range(ITERATIONS):
        for i in range(n):
            v = x[i]
            if J and rng.random() < J:
                ranked = np.argsort(completion)
                alt = int(ranked[M - 1 - int(np.where(ranked == v)[0][0])])
            else:
                j = rng.randrange(n)
                ri, rj = rng.random(), rng.random()
                alt = int(v + ri * FL * (x[j] - v)) if rj >= AP else rng.randrange(M)
            if alt == v:
                continue
            completion[v] -= rt[i, v]
            completion[alt] += rt[i, alt]
            candidate = completion.max()
            if candidate < makespan:
                makespan = candidate
                x[i] = alt
            else:
                completion[v] += rt[i, v]
                completion[alt] -= rt[i, alt]
    return makespan


def makespans(rt, population):
    n = population.shape[1]
    w = rt[np.arange(n)[None, :], population]
    return np.array([np.bincount(population[p], weights=w[p], minlength=M).max() for p in range(len(population))])


def ga(rt, seed_assignment, rng, budget, pop=50):
    n = rt.shape[0]
    P = rng.integers(0, M, (pop, n))
    P[0] = seed_assignment
    F = makespans(rt, P)
    remaining = budget - pop
    while remaining > 0:
        a = rng.integers(pop, size=(pop, 2))
        b = rng.integers(pop, size=(pop, 2))
        ia = np.where(F[a[:, 0]] <= F[a[:, 1]], a[:, 0], a[:, 1])
        ib = np.where(F[b[:, 0]] <= F[b[:, 1]], b[:, 0], b[:, 1])
        mask = rng.random((pop, n)) < 0.5
        C = np.where(mask, P[ia], P[ib])
        mutate = rng.random((pop, n)) < 2.0 / n
        C[mutate] = rng.integers(0, M, mutate.sum())
        FC = makespans(rt, C)
        remaining -= pop
        G, FG = np.vstack([P, C]), np.concatenate([F, FC])
        best = np.argsort(FG)[:pop]
        P, F = G[best], FG[best]
    return F.min()
