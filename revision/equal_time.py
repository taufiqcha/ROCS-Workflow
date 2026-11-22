"""Equal-evaluation and equal-time comparison on the 80 E2 instances.

Instances: E2 (Epigenomics, Montage, Genome, Cycles; 500 tasks; CCR 1 and 5; seeds 0 to 9; 16 VMs).
Methods, all seeded with the clustering schedules: ROCS, EZ+GA, EZ+PCSA, EZ+WS-SSA, EZ+LS.
Each method runs once to 500n evaluations. A wrapper around the makespan evaluation records every new
best value with its evaluation count and elapsed time, so the best makespan can be read at any
evaluation count (20n, 100n, 500n) and at any time. Equal time uses the time ROCS needs for 20n, 100n,
and 500n evaluations on the same instance.

Timing: 4 worker processes (the four performance cores of the Apple M1), each warmed up on one small
instance before timing so Numba compilation is excluded, and every timed run is repeated three times
for ROCS at 20n to report timing variability.
Run alone from the repository root: python revision/equal_time.py
"""
import argparse
import json
import os
import random
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
os.chdir(CODE)
import baselines  # noqa: E402
import model  # noqa: E402
import rocs as rocs_module  # noqa: E402
import comparators as a3  # noqa: E402
from baselines import chains, ga, pcsa  # noqa: E402

METHODS = ["ROCS", "EZ_GA", "EZ_PCSA", "EZ_WS_SSA", "EZ_LS"]
_raw = model.evaluate
TRACE = {"t0": 0.0, "count": 0, "best": np.inf, "log": []}


def traced(problem, x):
    out = _raw(problem, x)
    TRACE["count"] += 1
    if out[0] < TRACE["best"]:
        TRACE["best"] = out[0]
        TRACE["log"].append((TRACE["count"], time.perf_counter() - TRACE["t0"], out[0]))
    return out


for mod in (baselines, rocs_module, a3):
    mod.evaluate = traced


def start_trace():
    TRACE.update(t0=time.perf_counter(), count=0, best=np.inf, log=[])


def solve(method, problem, ez_seeds, seed, budget, chain, parents, m):
    if method == "ROCS":
        rocs_module.rocs(problem, ez_seeds[0].copy(), random.Random(seed), budget)
    elif method == "EZ_GA":
        rng = np.random.default_rng(seed)
        pop = rng.integers(0, m, (50, problem.exe.shape[0]))
        pop[:3] = ez_seeds
        ga(problem, pop, rng, budget)
    elif method == "EZ_PCSA":
        pcsa(problem, ez_seeds, np.random.default_rng(seed), budget, chain, parents)
    elif method == "EZ_WS_SSA":
        a3.ws_ssa(problem, np.random.default_rng(seed), budget, native=False, seeds=ez_seeds)
    else:
        a3.ez_ls(problem, ez_seeds[0].copy(), random.Random(seed), budget)


def warm_up():
    warnings.filterwarnings("ignore")
    runtime, edges = model.generate("Montage", 60, 7)
    wf = model.prepare(runtime, edges, 1.0, model.fleet(16))
    problem, _, ez_seeds, _ = model.heuristic_seeds(wf)
    chain, parents = chains(wf.exe.shape[0], wf.ptr, wf.parent)
    for mth in METHODS:
        solve(mth, problem, ez_seeds, 0, 200, chain, parents, 16)


def run(job):
    recipe, ccr, seed = job
    warnings.filterwarnings("ignore")
    runtime, edges = model.generate(recipe, 500, 1000 + seed)
    wf = model.prepare(runtime, edges, ccr, model.fleet(16))
    t = time.perf_counter()
    problem, base, ez_seeds, _ = model.heuristic_seeds(wf)
    heuristic_seconds = time.perf_counter() - t
    n, m = wf.exe.shape
    chain, parents = chains(n, wf.ptr, wf.parent)
    out = dict(workflow=recipe, ccr=ccr, seed=seed, n_tasks=n, best_heuristic=min(base.values()),
               heuristic_seconds=heuristic_seconds, traces={}, rocs20_seconds=[])
    for mth in METHODS:
        start_trace()
        solve(mth, problem, ez_seeds, seed, 500 * n, chain, parents, m)
        out["traces"][mth] = TRACE["log"] + [(TRACE["count"], time.perf_counter() - TRACE["t0"], TRACE["best"])]
    for _ in range(3):
        t = time.perf_counter()
        rocs_module.rocs(problem, ez_seeds[0].copy(), random.Random(seed), 20 * n)
        out["rocs20_seconds"].append(time.perf_counter() - t)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=str(HERE.parent / "results" / "revision" / "equal_time.jsonl"))
    a = p.parse_args()
    jobs = [(r, c, s) for r in ("Epigenomics", "Montage", "Genome", "Cycles") for c in (1.0, 5.0) for s in range(10)]
    with Pool(a.workers, initializer=warm_up) as pool, open(a.out, "w") as f:
        for done, row in enumerate(pool.imap_unordered(run, jobs), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
