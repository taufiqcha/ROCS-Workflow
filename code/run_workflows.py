"""Workflow benchmark E1, budget sensitivity E2, and fleet sensitivity E3 (Table 2).

Instances use seed 1000 + s, disjoint from the tuning seeds 0 to 9. HEFT, PEFT, and EZ are the
heuristics. PSO, CSA, CSA-OBL, GA, and PCSA are seeded with HEFT and PEFT. ROCS, EZ+GA, and EZ+PCSA
are seeded with the three best heuristic schedules.
"""
import argparse
import csv
import random
import time
import warnings
from multiprocessing import Pool

import numpy as np

from baselines import chains, csa, ga, pcsa, pso
from model import fleet, generate, heuristic_seeds, prepare
from rocs import rocs

RECIPES = ["Blast", "Bwa", "Cycles", "Epigenomics", "Genome", "Montage", "Seismology", "Soykb", "Srasearch"]
METHODS = ["HEFT", "PEFT", "EZ", "PSO", "CSA", "CSA_OBL", "GA", "PCSA", "ROCS", "EZ_GA", "EZ_PCSA"]


def run(job):
    recipe, size, ccr, seed, budget_factor, speeds = job
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, size, 1000 + seed)
    wf = prepare(runtime, edges, ccr, speeds)
    t = time.time()
    problem, result, ez_seeds, hp_seeds = heuristic_seeds(wf)
    seconds = {"EZ": time.time() - t}
    n, m = wf.exe.shape
    budget = budget_factor * n
    chain, parents = chains(n, wf.ptr, wf.parent)

    def timed(method, solve):
        t0 = time.time()
        result[method] = solve()
        seconds[method] = time.time() - t0

    def seeded_ga(seeds):
        rng = np.random.default_rng(seed)
        population = rng.integers(0, m, (50, n))
        population[:len(seeds)] = seeds
        return ga(problem, population, rng, budget)

    timed("PSO", lambda: pso(problem, hp_seeds, np.random.default_rng(seed), budget))
    timed("CSA", lambda: csa(problem, hp_seeds, np.random.default_rng(seed), budget))
    timed("CSA_OBL", lambda: csa(problem, hp_seeds, np.random.default_rng(seed), budget, obl=True))
    timed("GA", lambda: seeded_ga(hp_seeds))
    timed("PCSA", lambda: pcsa(problem, hp_seeds, np.random.default_rng(seed), budget, chain, parents))
    timed("ROCS", lambda: rocs(problem, ez_seeds[0].copy(), random.Random(seed), budget))
    timed("EZ_GA", lambda: seeded_ga(ez_seeds))
    timed("EZ_PCSA", lambda: pcsa(problem, ez_seeds, np.random.default_rng(seed), budget, chain, parents))
    return ([recipe, size, ccr, seed, budget_factor, n, len(edges), wf.lower_bound] + [result[k] for k in METHODS]
            + [seconds[k] for k in METHODS[2:]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipes", default=",".join(RECIPES))
    p.add_argument("--sizes", default="200,500,1000")
    p.add_argument("--ccr", default="0.1,1,5")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--budget", type=int, default=20, help="evaluations per task")
    p.add_argument("--vms", type=int, default=16)
    p.add_argument("--speeds", default="400,500,600")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", default="../results/e1.csv")
    a = p.parse_args()
    speeds = fleet(a.vms, tuple(int(s) for s in a.speeds.split(",")))
    jobs = [(r, n, float(c), s, a.budget, speeds) for n in map(int, a.sizes.split(",")) for s in range(a.seeds)
            for r in a.recipes.split(",") for c in a.ccr.split(",")]
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow", "size", "ccr", "seed", "budget_factor", "n_tasks", "n_edges", "lower_bound"]
                   + METHODS + ["seconds_" + k for k in METHODS[2:]])
        for done, row in enumerate(pool.imap(run, jobs), 1):
            w.writerow(row)
            f.flush()
            print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
