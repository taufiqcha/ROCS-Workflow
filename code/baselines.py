"""Compared metaheuristics on the VM-index encoding (Table 1): CSA, CSA-OBL, PSO, GA, and PCSA.

All methods decode with the shared priority order and receive the same evaluation budget.
Continuous positions in [0, M) are decoded by the floor function.
"""
import numpy as np

from model import evaluate
from rocs import opposite_rank


def chains(n, ptr, parent):
    """Linear chain containing each task, and the parent list of each task."""
    parents = [list(parent[ptr[i]:ptr[i + 1]]) for i in range(n)]
    children = [[] for _ in range(n)]
    for i in range(n):
        for a in parents[i]:
            children[a].append(i)
    chain = {}
    for i in range(n):
        members = [i]
        j = i
        while len(children[j]) == 1 and len(parents[children[j][0]]) == 1:
            j = children[j][0]
            members.append(j)
        j = i
        while len(parents[j]) == 1 and len(children[parents[j][0]]) == 1:
            j = parents[j][0]
            members.insert(0, j)
        chain[i] = members
    return chain, parents


class LocalMoves:
    """Chain move, pull to the heaviest parent, rank-opposition jump, or random move."""

    def __init__(self, problem, chain, parents, weights=(0.4, 0.2, 0.2, 0.2)):
        self.ptr, self.parent, self.comm = problem.ptr, problem.parent, problem.comm
        self.m = problem.exe.shape[1]
        self.chain, self.parents = chain, parents
        self.weights = np.array(weights, dtype=float) / sum(weights)

    def apply(self, x, busy, rng):
        i = int(rng.integers(len(x)))
        kind = rng.choice(4, p=self.weights)
        if kind == 0:
            members = self.chain[i]
            heads = self.parents[members[0]]
            x[members] = int(x[heads[rng.integers(len(heads))]]) if heads and rng.random() < 0.5 else int(rng.integers(self.m))
        elif kind == 1:
            if self.ptr[i + 1] > self.ptr[i]:
                e = self.ptr[i] + int(np.argmax(self.comm[self.ptr[i]:self.ptr[i + 1]]))
                x[i] = x[self.parent[e]]
        elif kind == 2:
            x[i] = opposite_rank(busy, x[i])
        else:
            x[i] = int(rng.integers(self.m))


def floor_eval(problem, position):
    return evaluate(problem, np.floor(position).astype(np.int64))[0]


def csa(problem, seeds, rng, budget, N=20, AP=0.1, FL=2.0, obl=False):
    """CSA of Askarzadeh (2016) with Eq. (4). With obl, every random crow is compared with its
    opposite M - x of Eq. (5) at initialization (CSA-OBL)."""
    n, m = problem.exe.shape
    X = rng.uniform(0, m, (N, n))
    evaluations = 0
    for k in range(len(seeds), N):
        if obl:
            opposite = m - 1e-9 - X[k]
            fa, fb = floor_eval(problem, X[k]), floor_eval(problem, opposite)
            evaluations += 2
            if fb < fa:
                X[k] = opposite
    for k, s in enumerate(seeds[:N]):
        X[k] = s + 0.5
    memory = X.copy()
    f_memory = np.array([floor_eval(problem, x) for x in X])
    evaluations += N
    while evaluations < budget:
        for i in range(N):
            j = int(rng.integers(N))
            if rng.random() >= AP:
                X[i] = X[i] + rng.random() * FL * (memory[j] - X[i])
            else:
                X[i] = rng.uniform(0, m, n)
            X[i] = np.clip(X[i], 0, m - 1e-9)
            f = floor_eval(problem, X[i])
            evaluations += 1
            if f <= f_memory[i]:
                memory[i], f_memory[i] = X[i].copy(), f
            if evaluations >= budget:
                break
    return f_memory.min()


def pso(problem, seeds, rng, budget, N=20, w=0.7, c1=1.5, c2=1.5):
    n, m = problem.exe.shape
    X = rng.uniform(0, m, (N, n))
    for k, s in enumerate(seeds[:N]):
        X[k] = s + 0.5
    V = rng.uniform(-1, 1, (N, n))
    P = X.copy()
    fP = np.array([floor_eval(problem, x) for x in X])
    evaluations = N
    g = int(np.argmin(fP))
    while evaluations < budget:
        r1, r2 = rng.random((N, n)), rng.random((N, n))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (P[g] - X)
        X = np.clip(X + V, 0, m - 1e-9)
        for k in range(N):
            f = floor_eval(problem, X[k])
            evaluations += 1
            if f <= fP[k]:
                P[k], fP[k] = X[k].copy(), f
            if evaluations >= budget:
                break
        g = int(np.argmin(fP))
    return fP.min()


def ga(problem, population, rng, budget):
    """Binary tournament, uniform crossover, reset mutation with rate 2/n, and (mu + lambda) survival."""
    pop, n = population.shape
    m = problem.exe.shape[1]
    P = population.copy()
    F = np.array([evaluate(problem, p)[0] for p in P])
    remaining = budget - pop
    while remaining > 0:
        children = np.empty_like(P)
        for c in range(pop):
            a, b = [min(rng.integers(pop, size=2), key=lambda z: F[z]) for _ in range(2)]
            mask = rng.random(n) < 0.5
            children[c] = np.where(mask, P[a], P[b])
            mutate = rng.random(n) < 2.0 / n
            children[c][mutate] = rng.integers(0, m, mutate.sum())
        F_children = np.array([evaluate(problem, p)[0] for p in children])
        remaining -= pop
        merged, F_merged = np.vstack([P, children]), np.concatenate([F, F_children])
        best = np.argsort(F_merged)[:pop]
        P, F = merged[best], F_merged[best]
    return F.min()


def pcsa(problem, seeds, rng, budget, chain, parents, N=50, AP=0.1, FL=1.0, p_local=0.2, random_share=0.25,
         flight=5):
    """Population CSA on the VM index. Each crow moves from its own memory toward the memory of a
    tournament-selected crow. A quarter of the non-seed crows are random with OBL by Eq. (5), and
    the others are perturbed seeds."""
    n, m = problem.exe.shape
    moves = LocalMoves(problem, chain, parents)
    evaluations = 0
    n_random = int(round(random_share * (N - len(seeds))))
    flock = []
    for k in range(N):
        if k < len(seeds):
            flock.append(seeds[k].copy())
        elif k >= len(seeds) + n_random:
            x = seeds[k % len(seeds)].copy()
            load = np.bincount(x, minlength=m).astype(float)
            for _ in range(int(rng.integers(1, 20))):
                moves.apply(x, load, rng)
            flock.append(x)
        else:
            x = rng.integers(0, m, n)
            opposite = (m - 1) - x
            fx, fo = evaluate(problem, x)[0], evaluate(problem, opposite)[0]
            evaluations += 2
            flock.append(x if fx <= fo else opposite)
    memory = np.array(flock)
    results = [evaluate(problem, x) for x in memory]
    evaluations += N
    f_memory = np.array([r[0] for r in results])
    busy = [r[1] for r in results]
    while evaluations < budget:
        for i in range(N):
            j, j2 = int(rng.integers(N)), int(rng.integers(N))
            j = j if f_memory[j] <= f_memory[j2] else j2
            x = memory[i].copy()
            r = rng.random()
            if r >= AP:
                mask = rng.random(n) < FL * r
                x[mask] = memory[j][mask]
            else:
                for _ in range(flight):
                    moves.apply(x, busy[i], rng)
            if rng.random() < p_local:
                moves.apply(x, busy[i], rng)
            f, busy[i] = evaluate(problem, x)
            evaluations += 1
            if f <= f_memory[i]:
                memory[i], f_memory[i] = x.copy(), f
            if evaluations >= budget:
                break
    return f_memory.min()
