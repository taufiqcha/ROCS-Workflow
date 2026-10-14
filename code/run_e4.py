"""E4: optimality gap on 36 small instances with four VMs (Table 2 and Section 5.4).

The smallest WfCommons instances of six applications use seed 2000 + s. Every method receives 20n
evaluations, and DCSA is also run at 100n, 500n, and 2000n.
"""
import argparse
import csv
import random
import time
import warnings
from multiprocessing import Pool

import numpy as np

import exact
from baselines import chains, csa, ga, pcsa
from dcsa import DualDecoder, dcsa, dual_seeds
from model import fleet, generate, heuristic_seeds, prepare
from rocs import rocs

SIZES = {"Blast": 45, "Epigenomics": 43, "Genome": 54, "Montage": 60, "Srasearch": 24, "Cycles": 69}
SPEEDS = fleet(4)
METHODS = ["HEFT", "PEFT", "EZ", "CSA_OBL", "GA", "ROCS", "EZ_GA", "EZ_PCSA"]
DCSA_BUDGETS = [20, 100, 500, 2000]


def instance(recipe, ccr, seed):
    runtime, edges = generate(recipe, SIZES[recipe], 2000 + seed)
    wf = prepare(runtime, edges, ccr, SPEEDS)
    return wf, len(edges), heuristic_seeds(wf)


def heuristics_and_metaheuristics(recipe, ccr, seed):
    wf, n_edges, (problem, result, ez_seeds, hp_seeds) = instance(recipe, ccr, seed)
    n, m = wf.exe.shape
    budget = 20 * n
    chain, parents = chains(n, wf.ptr, wf.parent)
    rng = lambda: np.random.default_rng(seed)
    hp_population = rng().integers(0, m, (50, n))
    hp_population[:2] = hp_seeds
    ez_population = rng().integers(0, m, (50, n))
    ez_population[:3] = ez_seeds
    result["CSA_OBL"] = csa(problem, hp_seeds, rng(), budget, obl=True)
    result["GA"] = ga(problem, hp_population, rng(), budget)
    result["ROCS"] = rocs(problem, ez_seeds[0].copy(), random.Random(seed), budget)
    result["EZ_GA"] = ga(problem, ez_population, rng(), budget)
    result["EZ_PCSA"] = pcsa(problem, ez_seeds, rng(), budget, chain, parents)
    return wf, n_edges, ez_seeds, result


def run_dcsa(job):
    recipe, ccr, seed, budget_factor = job
    warnings.filterwarnings("ignore")
    wf, _, (problem, _, ez_seeds, _) = instance(recipe, ccr, seed)
    n = wf.exe.shape[0]
    chain, parents = chains(n, wf.ptr, wf.parent)
    decoder = DualDecoder(wf)
    best, _, _ = dcsa(decoder, dual_seeds(wf, problem, ez_seeds), np.random.default_rng(seed), budget_factor * n,
                      chain, parents)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=120)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", default="../results/e4.csv")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    keys = [(r, c, s) for r in SIZES for c in (1.0, 5.0) for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        dual = pool.map(run_dcsa, [k + (b,) for k in keys for b in DCSA_BUDGETS])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow", "ccr", "seed", "n_tasks", "n_edges", "cpsat_status", "cpsat_best", "cpsat_bound",
                    "cpsat_seconds"] + METHODS + [f"DCSA_{b}" for b in DCSA_BUDGETS])
        for q, (recipe, ccr, seed) in enumerate(keys):
            wf, n_edges, ez_seeds, result = heuristics_and_metaheuristics(recipe, ccr, seed)
            t = time.time()
            best, bound, status = exact.solve(wf.exe, wf.ptr, wf.parent, wf.comm, ez_seeds[0], a.time_limit)
            row = ([recipe, ccr, seed, wf.exe.shape[0], n_edges, status, best, bound, time.time() - t]
                   + [result[k] for k in METHODS] + dual[q * len(DCSA_BUDGETS):(q + 1) * len(DCSA_BUDGETS)])
            w.writerow(row)
            f.flush()
            print(row[:9], flush=True)


if __name__ == "__main__":
    main()
