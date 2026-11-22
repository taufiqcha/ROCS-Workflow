"""A recent discrete comparator and a strong seed-refinement baseline on the E1 instances.

WS-SSA follows Sharawy et al., Sci Rep 16:13402 (2026), https://doi.org/10.1038/s41598-026-48037-w:
- integer task-to-VM positions, every task on a uniformly random VM at initialization;
- the food source F is the best position found so far;
- leader (first salp), per task: x = F + c1((ub - lb)c2 + lb) if c3 >= 0.5, else F - c1((ub - lb)c2 + lb), with c1 = 2 exp(-(4t/T)^2) and c2, c3 uniform in [0, 1];
- followers: x_i = (x_i + x_{i-1}) / 2, with x_{i-1} the preceding salp;
- population 50 (the paper's default). The paper does not state the rounding rule, so positions are
  rounded to the nearest integer and clipped to [lb, ub] = [0, m - 1].
- Native decoder: tasks in a topological order, each starts at the later of its data-ready
  time and the finish of the last task on its VM (no insertion). The shared priority order of the
  benchmark is used as the topological order, and the transfer model is the benchmark's.
WS-SSA-SD is the same search with the shared insertion decoder of the benchmark, labelled as a standardised
shared-decoder variant. EZ+WS-SSA is WS-SSA-SD whose first three salps are the clustering seeds, as for
EZ+GA, so the search operator can be compared from the same starting schedules.

EZ+LS is a seed-refinement baseline: from the ROCS seed, a random task is moved to the VM that gives
the shortest makespan among all other VMs (each trial decode counted), and the move is kept when it is
not worse. Every method receives 20n makespan evaluations, the E1 budget.

Run from the repository root: python revision/comparators.py [--workers 7]
"""
import argparse
import csv
import math
import os
import random
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
os.chdir(CODE)
from model import evaluate, fleet, generate, heuristic_seeds, prepare  # noqa: E402

RECIPES = ["Blast", "Bwa", "Cycles", "Epigenomics", "Genome", "Montage", "Seismology", "Soykb", "Srasearch"]
METHODS = ["WS_SSA", "WS_SSA_SD", "EZ_WS_SSA", "EZ_LS"]


@njit(cache=True)
def native_decode(exe, order, ptr, parent, comm, x):
    """Start-time rule of WS-SSA: ST = max(data-ready time, availability of the VM), no insertion."""
    n, m = exe.shape
    available = np.zeros(m)
    done = np.zeros(n)
    for q in range(n):
        i = order[q]
        k = x[i]
        est = available[k]
        for e in range(ptr[i], ptr[i + 1]):
            a = parent[e]
            t = done[a] + (comm[e] if x[a] != k else 0.0)
            if t > est:
                est = t
        done[i] = est + exe[i, k]
        available[k] = done[i]
    return done.max()


def ws_ssa(problem, rng, budget, native=True, pop=50, seeds=()):
    n, m = problem.exe.shape
    lb, ub = 0, m - 1
    if native:
        fitness = lambda x: native_decode(problem.exe, problem.order, problem.ptr, problem.parent, problem.comm, x)
    else:
        fitness = lambda x: evaluate(problem, x)[0]
    X = rng.integers(lb, ub + 1, (pop, n))
    for k, s in enumerate(seeds):       # EZ+WS-SSA: the clustering seeds replace the first random salps
        X[k] = s
    T = max(1, budget // pop)           # iterations allowed by the evaluation budget
    best, food = math.inf, None
    used, t = 0, 0
    while used < budget:
        t += 1
        for s in range(pop):            # WS-SSA: evaluate every salp, update the food source
            if used >= budget:
                break
            f = fitness(X[s])
            used += 1
            if f < best:
                best, food = f, X[s].copy()
        c1 = 2.0 * math.exp(-(4.0 * t / T) ** 2)
        step = c1 * ((ub - lb) * rng.random(n) + lb)
        X[0] = np.where(rng.random(n) >= 0.5, food + step, food - step).round()
        X[1:] = ((X[1:] + X[:-1]) / 2.0).round()   # each follower averages with the salp ahead of it
        X = np.clip(X, lb, ub).astype(np.int64)
    return best


def ez_ls(problem, x, rng, budget):
    n, m = problem.exe.shape
    makespan = evaluate(problem, x)[0]
    used = 0
    while used < budget:
        i = rng.randrange(n)
        v = x[i]
        best, alt = math.inf, v
        for k in range(m):
            if k == v or used >= budget:
                continue
            x[i] = k
            mk = evaluate(problem, x)[0]
            used += 1
            if mk < best:
                best, alt = mk, k
        if best <= makespan:
            x[i], makespan = alt, best
        else:
            x[i] = v
    return makespan


def run(job):
    recipe, size, ccr, seed = job
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, size, 1000 + seed)
    wf = prepare(runtime, edges, ccr, fleet(16))
    problem, base, ez_seeds, _ = heuristic_seeds(wf)
    n = wf.exe.shape[0]
    budget = 20 * n
    row = [recipe, size, ccr, seed, n, min(base.values())]
    for method in METHODS:
        t = time.time()
        if method == "WS_SSA":
            mk = ws_ssa(problem, np.random.default_rng(seed), budget, native=True)
        elif method == "WS_SSA_SD":
            mk = ws_ssa(problem, np.random.default_rng(seed), budget, native=False)
        elif method == "EZ_WS_SSA":
            mk = ws_ssa(problem, np.random.default_rng(seed), budget, native=False, seeds=ez_seeds)
        else:
            mk = ez_ls(problem, ez_seeds[0].copy(), random.Random(seed), budget)
        row += [mk, time.time() - t]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out", default=str(HERE.parent / "results" / "revision" / "comparators.csv"))
    a = p.parse_args()
    jobs = [(r, n, c, s) for n in (200, 500, 1000) for s in range(a.seeds) for r in RECIPES for c in (0.1, 1.0, 5.0)]
    head = ["workflow", "size", "ccr", "seed", "n_tasks", "best_heuristic"]
    for mth in METHODS:
        head += [mth, "seconds_" + mth]
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for done, row in enumerate(pool.imap_unordered(run, jobs), 1):
            w.writerow(row)
            f.flush()
            if done % 50 == 0:
                print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
