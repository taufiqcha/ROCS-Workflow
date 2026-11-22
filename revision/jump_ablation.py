"""Ablation of rank-opposition jumping on the 2,430 E1 instances.

Every variant starts from the same seed (the best of HEFT, PEFT, and EZ), uses the same priority
order, the same random stream random.Random(seed), and the budget 20n of E1. Only the destination of
the jump branch, taken with probability J = 0.3, changes:

- ROCS: VM with the opposite busy-time rank, as in rocs.py.
- J0: no jump (J = 0), so every move is a crow move.
- UNIFORM: VM drawn uniformly at random.
- LEAST: VM with the smallest busy time.
- EFT: VM that minimises the finish time of the task when only that task moves (m decodes per jump,
  each counted in the budget; the accepted move re-decodes an already evaluated assignment).

Run from the repository root: python revision/jump_ablation.py [--workers 7]
"""
import argparse
import csv
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
import os  # noqa: E402
os.chdir(CODE)
from model import evaluate, fleet, generate, heuristic_seeds, prepare  # noqa: E402
from rocs import opposite_rank  # noqa: E402

RECIPES = ["Blast", "Bwa", "Cycles", "Epigenomics", "Genome", "Montage", "Seismology", "Soykb", "Srasearch"]
VARIANTS = ["ROCS", "J0", "UNIFORM", "LEAST", "EFT"]


def search(problem, x, rng, budget, variant, J=0.3, AP=0.1, FL=0.5):
    """ROCS with the destination of the jump branch replaced by the variant."""
    n, m = problem.exe.shape
    makespan, busy = evaluate(problem, x)
    used, jumps, accepted = 0, 0, 0
    while used < budget:
        i = rng.randrange(n)
        v = x[i]
        if variant != "J0" and rng.random() < J:
            jumps += 1
            if variant == "ROCS":
                alt = opposite_rank(busy, v)
            elif variant == "UNIFORM":
                alt = rng.randrange(m)
            elif variant == "LEAST":
                alt = int(np.argmin(busy))
            else:  # EFT: best destination for task i, every trial decode counts
                best, alt = None, v
                for k in range(m):
                    if k == v or used >= budget:
                        continue
                    x[i] = k
                    mk, _ = evaluate(problem, x)
                    used += 1
                    if best is None or mk < best:
                        best, alt = mk, k
                x[i] = v
        else:
            j = rng.randrange(n)
            if rng.random() >= AP:
                alt = int(v + rng.random() * FL * (x[j] - v))
            else:
                alt = rng.randrange(m)
        if variant != "EFT":
            used += 1
        if alt == v:
            continue
        x[i] = alt
        candidate, candidate_busy = evaluate(problem, x)
        if candidate <= makespan:
            accepted += candidate < makespan
            makespan, busy = candidate, candidate_busy
        else:
            x[i] = v
    return makespan, jumps, accepted


def run(job):
    recipe, size, ccr, seed = job
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, size, 1000 + seed)
    wf = prepare(runtime, edges, ccr, fleet(16))
    problem, base, ez_seeds, _ = heuristic_seeds(wf)
    n = wf.exe.shape[0]
    row = [recipe, size, ccr, seed, n, min(base.values())]
    for variant in VARIANTS:
        t = time.time()
        mk, jumps, improved = search(problem, ez_seeds[0].copy(), random.Random(seed), 20 * n, variant)
        row += [mk, time.time() - t, improved]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out", default=str(HERE.parent / "results" / "revision" / "jump_ablation.csv"))
    a = p.parse_args()
    jobs = [(r, n, c, s) for n in (200, 500, 1000) for s in range(a.seeds) for r in RECIPES for c in (0.1, 1.0, 5.0)]
    head = ["workflow", "size", "ccr", "seed", "n_tasks", "best_heuristic"]
    for v in VARIANTS:
        head += [v, "seconds_" + v, "improving_moves_" + v]
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
