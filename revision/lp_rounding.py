"""Does a schedule better than LPT exist on Rnaseq? Round the LP of independent_bounds.py.

The LP gives, for every task, fractions over the three speed groups. Each task is assigned to the
group with its largest fraction, and LPT schedules the tasks of each group on the VMs of that group.
The result is a feasible schedule, so its makespan is an upper bound on the optimum.
Run from the repository root: PYTHONHASHSEED=0 python revision/lp_rounding.py
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import independent_bounds as A  # noqa: E402  (changes cwd to upload_github/code)

ind = A.ind


def lp_assignment(length):
    n, G = len(length), len(A.GROUPS)
    r = length[:, None] / A.GROUPS[None, :] + A.C
    cost = np.zeros(n * G + 1); cost[-1] = 1.0
    rows = np.repeat(np.arange(G), n); cols = (np.arange(n)[None, :] * G + np.arange(G)[:, None]).ravel()
    A_ub = coo_matrix((np.r_[r.T.ravel(), -A.COUNT], (np.r_[rows, np.arange(G)], np.r_[cols, np.full(G, n * G)])),
                      shape=(G, n * G + 1)).tocsr()
    A_eq = coo_matrix((np.ones(n * G), (np.repeat(np.arange(n), G), np.arange(n * G))), shape=(n, n * G + 1)).tocsr()
    sol = linprog(cost, A_ub=A_ub, b_ub=np.zeros(G), A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    return sol.x[:-1].reshape(n, G).argmax(1)


def rounded_schedule(length):
    group = lp_assignment(length)
    busy = np.zeros(ind.M)
    for g, speed in enumerate(A.GROUPS):
        vms = np.where(ind.SPEEDS == speed)[0]
        tasks = np.where(group == g)[0]
        for i in tasks[np.argsort(-length[tasks])]:
            k = vms[np.argmin(busy[vms])]
            busy[k] += length[i] / speed + A.C
    return busy.max()


def job(key):
    dataset, size, seed = key
    return dict(dataset=dataset, size=size, seed=seed, lp_rounded=rounded_schedule(ind.lengths(dataset, size, seed)))


if __name__ == "__main__":
    b = pd.read_csv(HERE.parent / "results" / "revision" / "independent_bounds.csv")
    keys = list(b[["dataset", "size", "seed"]].itertuples(index=False, name=None))
    with Pool(7) as pool:
        rows = pool.map(job, keys)
    d = b.merge(pd.DataFrame(rows), on=["dataset", "size", "seed"])
    assert (d.lp_rounded >= d.lb_lp - 1e-6).all()
    d.to_csv(HERE.parent / "results" / "revision" / "independent_bounds.csv", index=False)
    print("ok")
