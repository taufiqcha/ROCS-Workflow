"""E5: independent tasks on 54 VMs, three datasets, five sizes, and 30 seeds (Table 3).

CSA, CSA-OBL, and CSA-ROJ start from the same random assignment, CSA-OBL from its opposite by
Eq. (5). GA and ROCS start from LPT. GA receives 20n evaluations.
"""
import argparse
import csv
import os
import random
import warnings
from multiprocessing import Pool

import numpy as np

import independent as ind

METHODS = ["MIN_MIN", "LPT", "CSA", "CSA_OBL", "CSA_ROJ", "GA", "ROCS_LPT"]


def run(job):
    dataset, size, seed = job
    warnings.filterwarnings("ignore")
    length = ind.lengths(dataset, size, seed)
    rt = ind.run_times(length)
    n = len(length)
    r0 = random.Random(seed)
    start = np.array([r0.randrange(ind.M) for _ in range(n)])
    stream = lambda k: random.Random(seed * 101 + k)
    lpt_makespan, x_lpt = ind.lpt(rt, length)
    result = {
        "MIN_MIN": ind.min_min(rt, length)[0],
        "LPT": lpt_makespan,
        "CSA": ind.crow_search(rt, start.copy(), stream(1)),
        "CSA_OBL": ind.crow_search(rt, (ind.M - 1) - start, stream(2)),
        "CSA_ROJ": ind.crow_search(rt, start.copy(), stream(5), J=0.3),
        "GA": ind.ga(rt, x_lpt, np.random.default_rng(seed), 20 * n),
        "ROCS_LPT": ind.crow_search(rt, x_lpt.copy(), stream(7), J=0.3),
    }
    return [dataset, size, seed, n, ind.lower_bound(length)] + [result[k] for k in METHODS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--datasets", default="simple,stratified,rnaseq")
    p.add_argument("--out", default="../results/e5.csv")
    a = p.parse_args()
    # WfCommons orders the Rnaseq tasks through a set, so the hash seed fixes the task order.
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run as: PYTHONHASHSEED=0 python run_e5.py")
    jobs = [(d, n, s) for d in a.datasets.split(",") for n in (1000, 3000, 5000, 7000, 10000)
            for s in range(a.seeds)]
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "size", "seed", "n_tasks", "lower_bound"] + METHODS)
        for done, row in enumerate(pool.imap(run, jobs), 1):
            w.writerow(row)
            f.flush()
            print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
