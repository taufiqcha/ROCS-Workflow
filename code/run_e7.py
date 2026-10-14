"""E7: dual encoding at a budget of 100n on the E1 instances where refinement pays off (Table 5).

All methods are seeded with the EZ, HEFT, and PEFT schedules. The dual-encoded methods also receive
the keys of the shared, HEFT, and PEFT orders.
"""
import argparse
import csv
import random
import time
import warnings
from multiprocessing import Pool

import numpy as np

from baselines import chains, ga, pcsa
from dcsa import DualDecoder, dcsa, dual_seeds, ga_dual
from model import fleet, generate, heuristic_seeds, prepare
from rocs import rocs

SPEEDS = fleet(16)
METHODS = ["EZ", "ROCS", "EZ_GA", "EZ_PCSA", "GA_DUAL", "DCSA", "DCSA_NO_OBL", "DCSA_NO_KEYS"]


def run(job):
    recipe, size, ccr, seed, budget_factor = job
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, size, 1000 + seed)
    wf = prepare(runtime, edges, ccr, SPEEDS)
    problem, heuristics, ez_seeds, _ = heuristic_seeds(wf)
    n, m = wf.exe.shape
    budget = budget_factor * n
    chain, parents = chains(n, wf.ptr, wf.parent)
    seeds = dual_seeds(wf, problem, ez_seeds)
    result, seconds = {"EZ": min(heuristics.values())}, {}

    def timed(method, solve):
        t = time.time()
        result[method] = solve()
        seconds[method] = time.time() - t

    def dual(method, **options):
        return method(DualDecoder(wf), seeds, np.random.default_rng(seed), budget, **options)

    rng = lambda: np.random.default_rng(seed)
    population = rng().integers(0, m, (50, n))
    population[:3] = ez_seeds
    timed("ROCS", lambda: rocs(problem, ez_seeds[0].copy(), random.Random(seed), budget))
    timed("EZ_GA", lambda: ga(problem, population, rng(), budget))
    timed("EZ_PCSA", lambda: pcsa(problem, ez_seeds, rng(), budget, chain, parents))
    timed("GA_DUAL", lambda: dual(ga_dual))
    timed("DCSA", lambda: dual(dcsa, chain=chain, parents=parents)[0])
    timed("DCSA_NO_OBL", lambda: dual(dcsa, chain=chain, parents=parents, obl=False)[0])
    timed("DCSA_NO_KEYS", lambda: dual(dcsa, chain=chain, parents=parents, p_key=0.0, sigma=0.0)[0])
    return ([recipe, size, ccr, seed, budget_factor, n, wf.lower_bound] + [result[k] for k in METHODS]
            + [seconds[k] for k in METHODS[1:]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--budget", type=int, default=100, help="evaluations per task")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--out", default="../results/e7.csv")
    a = p.parse_args()
    jobs = [(r, n, c, s, a.budget) for n in (200, 500) for r in ("Epigenomics", "Montage", "Genome", "Cycles")
            for c in (1.0, 5.0) for s in range(a.seeds)]
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow", "size", "ccr", "seed", "budget_factor", "n_tasks", "lower_bound"] + METHODS
                   + ["seconds_" + k for k in METHODS[1:]])
        for done, row in enumerate(pool.imap(run, jobs), 1):
            w.writerow(row)
            f.flush()
            print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
