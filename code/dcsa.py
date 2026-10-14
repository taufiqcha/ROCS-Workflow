"""Dual-encoded crow search DCSA (Algorithm 2) and the dual-encoded GA (GA-dual).

A solution is a VM assignment x and a continuous priority key per task. The decoder repeatedly
schedules the ready task with the largest key on VM x[i] with insertion, so every decoded order
respects the dependencies.
"""
import numpy as np
from numba import njit

from baselines import LocalMoves
from model import Problem, insertion_slot


def keys_from_order(order):
    """Keys that make the decoder reproduce a given priority order."""
    keys = np.empty(len(order))
    keys[order] = len(order) - np.arange(len(order))
    return keys


def child_lists(n, ptr, parent):
    children = [[] for _ in range(n)]
    for i in range(n):
        for e in range(ptr[i], ptr[i + 1]):
            children[parent[e]].append(i)
    c_ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        c_ptr[i + 1] = c_ptr[i] + len(children[i])
    child = np.array([j for i in range(n) for j in children[i]], dtype=np.int64)
    return c_ptr, child


@njit(cache=True)
def decode_keys(exe, keys, ptr, parent, comm, c_ptr, child, x):
    n, m = exe.shape
    start = np.zeros((m, n))
    finish = np.zeros((m, n))
    count = np.zeros(m, dtype=np.int64)
    done = np.zeros(n)
    waiting = np.empty(n, dtype=np.int64)
    ready = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        waiting[i] = ptr[i + 1] - ptr[i]
        ready[i] = waiting[i] == 0
    for _ in range(n):
        i, best = -1, -1e300
        for t in range(n):
            if ready[t] and keys[t] > best:
                best, i = keys[t], t
        ready[i] = False
        k = x[i]
        est = 0.0
        for e in range(ptr[i], ptr[i + 1]):
            a = parent[e]
            t = done[a] + (comm[e] if x[a] != k else 0.0)
            if t > est:
                est = t
        s, pos = insertion_slot(start[k], finish[k], count[k], est, exe[i, k])
        for q in range(count[k], pos, -1):
            start[k, q] = start[k, q - 1]
            finish[k, q] = finish[k, q - 1]
        start[k, pos] = s
        finish[k, pos] = s + exe[i, k]
        count[k] += 1
        done[i] = s + exe[i, k]
        for e in range(c_ptr[i], c_ptr[i + 1]):
            j = child[e]
            waiting[j] -= 1
            if waiting[j] == 0:
                ready[j] = True
    busy = np.zeros(m)
    for k in range(m):
        for q in range(count[k]):
            busy[k] += finish[k, q] - start[k, q]
    return done.max(), busy


class DualDecoder:
    """Key-based decoder that counts the evaluations used."""

    def __init__(self, wf):
        self.exe, self.ptr, self.parent, self.comm = wf.exe, wf.ptr, wf.parent, wf.comm
        self.c_ptr, self.child = child_lists(wf.exe.shape[0], wf.ptr, wf.parent)
        self.evaluations = 0

    def __call__(self, x, keys):
        self.evaluations += 1
        return decode_keys(self.exe, keys, self.ptr, self.parent, self.comm, self.c_ptr, self.child, x.copy())


def dual_seeds(wf, problem, ez_seeds):
    """The EZ seeds with the keys of the shared order, plus the best seed with the HEFT and PEFT keys."""
    k0 = keys_from_order(problem.order)
    return [(x, k0) for x in ez_seeds] + [(ez_seeds[0], keys_from_order(wf.heft_order)),
                                          (ez_seeds[0], keys_from_order(wf.peft_order))]


def dcsa(decoder, seeds, rng, budget, chain, parents, N=50, AP=0.1, FL=1.0, p_local=0.2, p_key=0.2,
         seed_share=0.25, obl=True, sigma=0.05, flight=5):
    """Algorithm 2. Returns the best makespan, assignment, and keys."""
    n, m = decoder.exe.shape
    moves = LocalMoves(Problem(decoder.exe, None, decoder.ptr, decoder.parent, decoder.comm), chain, parents)
    scale = max(np.ptp(seeds[0][1]), 1e-9)
    X, K = [], []
    n_other = N - len(seeds)
    n_random = int(round((1 - seed_share) * n_other))
    for q in range(N):
        if q < len(seeds):
            x, k = seeds[q][0].copy(), seeds[q][1].copy()
        elif q < len(seeds) + n_other - n_random:
            x, k = seeds[q % len(seeds)][0].copy(), seeds[q % len(seeds)][1].copy()
            load = np.bincount(x, minlength=m).astype(float)
            for _ in range(int(rng.integers(1, 20))):
                moves.apply(x, load, rng)
            k = k + rng.normal(0, sigma * scale, n)
        else:
            x = rng.integers(0, m, n)
            k = rng.uniform(0, scale, n)
            if obl:
                x_opp, k_opp = (m - 1) - x, k.min() + k.max() - k
                if decoder(x_opp, k_opp)[0] < decoder(x, k)[0]:
                    x, k = x_opp, k_opp
        X.append(x)
        K.append(k)
    X, K = np.array(X), np.array(K)
    results = [decoder(X[q], K[q]) for q in range(N)]
    F = np.array([r[0] for r in results])
    busy = [r[1] for r in results]
    while decoder.evaluations < budget:
        for i in range(N):
            j, j2 = int(rng.integers(N)), int(rng.integers(N))
            j = j if F[j] <= F[j2] else j2
            x, k = X[i].copy(), K[i].copy()
            r = rng.random()
            if r >= AP:
                mask = rng.random(n) < FL * r / 2
                x[mask] = X[j][mask]
                k = k + r * FL * (K[j] - k)
            else:
                for _ in range(flight):
                    moves.apply(x, busy[i], rng)
                k = k + rng.normal(0, sigma * scale, n)
            if rng.random() < p_local:
                moves.apply(x, busy[i], rng)
            if rng.random() < p_key:
                a, b = rng.integers(n, size=2)
                k[a] = k[b] + rng.normal(0, 1e-3 * scale)
            f, busy[i] = decoder(x, k)
            if f <= F[i]:
                X[i], K[i], F[i] = x, k, f
            if decoder.evaluations >= budget:
                break
    best = int(np.argmin(F))
    return F[best], X[best], K[best]


def ga_dual(decoder, seeds, rng, budget, pop=50):
    """GA operators of baselines.ga on both chromosomes, with Gaussian key mutation at rate 2/n."""
    n, m = decoder.exe.shape
    scale = max(np.ptp(seeds[0][1]), 1e-9)
    X = rng.integers(0, m, (pop, n))
    K = rng.uniform(0, scale, (pop, n))
    for q, (x, k) in enumerate(seeds[:pop]):
        X[q], K[q] = x, k
    F = np.array([decoder(X[q], K[q])[0] for q in range(pop)])
    while decoder.evaluations < budget:
        CX, CK = np.empty_like(X), np.empty_like(K)
        for c in range(pop):
            a, b = [min(rng.integers(pop, size=2), key=lambda z: F[z]) for _ in range(2)]
            mask = rng.random(n) < 0.5
            CX[c] = np.where(mask, X[a], X[b])
            CK[c] = np.where(mask, K[a], K[b])
            mutate = rng.random(n) < 2.0 / n
            CX[c][mutate] = rng.integers(0, m, mutate.sum())
            mutate_keys = rng.random(n) < 2.0 / n
            CK[c][mutate_keys] += rng.normal(0, 0.05 * scale, mutate_keys.sum())
        FC = np.array([decoder(CX[c], CK[c])[0] for c in range(pop)])
        GX, GK, GF = np.vstack([X, CX]), np.vstack([K, CK]), np.concatenate([F, FC])
        best = np.argsort(GF)[:pop]
        X, K, F = GX[best], GK[best], GF[best]
    return F.min()
