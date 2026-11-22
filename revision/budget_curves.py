"""Matched-budget curves on the 36 small E4 instances.

For budgets of 20n, 100n, 500n, and 2000n evaluations, every method starts from the same seeds:
- DCSA: searches the VM assignment and the priority keys, as in E4;
- DCSA-FIXED: the same algorithm with key moves and key perturbation switched off (p_key = 0,
  sigma = 0), so the priority order stays that of the seeds;
- GA-DUAL: the dual-encoded GA;
- EZ+PCSA and ROCS: VM-only methods on the shared priority order.
The DCSA column reproduces DCSA_20 to DCSA_2000 of e4.csv exactly (checked in the analysis).

Run from the repository root: python revision/budget_curves.py [--workers 7]
"""
import argparse
import csv
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
os.chdir(CODE)
from baselines import chains, pcsa  # noqa: E402
from dcsa import DualDecoder, dcsa, dual_seeds, ga_dual  # noqa: E402
from rocs import rocs  # noqa: E402
from run_e4 import SIZES, instance  # noqa: E402

BUDGETS = [20, 100, 500, 2000]
METHODS = ["DCSA", "DCSA_FIXED", "GA_DUAL", "EZ_PCSA", "ROCS"]


def run(job):
    recipe, ccr, seed, factor, method = job
    warnings.filterwarnings("ignore")
    wf, _, (problem, _, ez_seeds, _) = instance(recipe, ccr, seed)
    n = wf.exe.shape[0]
    budget = factor * n
    chain, parents = chains(n, wf.ptr, wf.parent)
    t = time.time()
    if method == "DCSA":
        mk = dcsa(DualDecoder(wf), dual_seeds(wf, problem, ez_seeds), np.random.default_rng(seed), budget, chain, parents)[0]
    elif method == "DCSA_FIXED":
        mk = dcsa(DualDecoder(wf), dual_seeds(wf, problem, ez_seeds), np.random.default_rng(seed), budget, chain, parents,
                  p_key=0.0, sigma=0.0)[0]
    elif method == "GA_DUAL":
        mk = ga_dual(DualDecoder(wf), dual_seeds(wf, problem, ez_seeds), np.random.default_rng(seed), budget)
    elif method == "EZ_PCSA":
        mk = pcsa(problem, ez_seeds, np.random.default_rng(seed), budget, chain, parents)
    else:
        mk = rocs(problem, ez_seeds[0].copy(), random.Random(seed), budget)
    return [recipe, ccr, seed, n, factor, method, mk, time.time() - t]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", default=str(HERE.parent / "results" / "revision" / "budget_curves.csv"))
    a = p.parse_args()
    jobs = [(r, c, s, b, m) for r in SIZES for c in (1.0, 5.0) for s in range(3) for b in BUDGETS for m in METHODS]
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow", "ccr", "seed", "n_tasks", "budget_factor", "method", "makespan", "seconds"])
        for done, row in enumerate(pool.imap_unordered(run, jobs), 1):
            w.writerow(row)
            f.flush()
            if done % 100 == 0:
                print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
