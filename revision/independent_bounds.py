"""Valid lower bounds for the independent-task model.

Task i on VM k occupies r_ik = l_i / s_k + c with c = 300 MB / 1000 MB/s, and the makespan is the
largest VM busy time. Two valid bounds are computed for every E5 instance:

- LB_min: (sum_i l_i + n c s_min) / sum_k s_k. Summing s_k * C >= s_k * B_k over the VMs gives
  sum_k s_k C >= sum_i (l_i + c s_k(i)) >= sum_i l_i + n c s_min.
- LB_LP: the optimum of the fractional assignment LP min C s.t. sum_i x_ik r_ik <= C for every VM,
  sum_k x_ik = 1, x >= 0, combined with max_i (l_i / s_max + c). Any integer schedule is feasible
  for the LP with C equal to its makespan, so LB_LP <= OPT. VMs of equal speed are merged into one
  group with capacity (group size) * C, which leaves the optimum unchanged.

Run from the repository root: PYTHONHASHSEED=0 python revision/independent_bounds.py
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
CODE = HERE.parent / "code"
RES = HERE.parent / "results"
sys.path.insert(0, str(CODE))
os.chdir(CODE)
import independent as ind  # noqa: E402

C = ind.TRANSFER
GROUPS = np.unique(ind.SPEEDS)
COUNT = np.array([(ind.SPEEDS == g).sum() for g in GROUPS], dtype=float)


def lb_min(length):
    return (length.sum() + len(length) * C * ind.SPEEDS.min()) / ind.SPEEDS.sum()


def lb_lp(length):
    n, G = len(length), len(GROUPS)
    r = length[:, None] / GROUPS[None, :] + C                      # n x G
    # variables: y_ig (n*G) then C; minimise C
    cost = np.zeros(n * G + 1); cost[-1] = 1.0
    # capacity: sum_i r_ig y_ig - count_g * C <= 0
    rows = np.repeat(np.arange(G), n); cols = (np.arange(n)[None, :] * G + np.arange(G)[:, None]).ravel()
    vals = r.T.ravel()
    A_ub = coo_matrix((np.r_[vals, -COUNT], (np.r_[rows, np.arange(G)], np.r_[cols, np.full(G, n * G)])),
                      shape=(G, n * G + 1)).tocsr()
    # assignment: sum_g y_ig = 1
    A_eq = coo_matrix((np.ones(n * G), (np.repeat(np.arange(n), G), np.arange(n * G))), shape=(n, n * G + 1)).tocsr()
    sol = linprog(cost, A_ub=A_ub, b_ub=np.zeros(G), A_eq=A_eq, b_eq=np.ones(n), bounds=(0, None), method="highs")
    assert sol.status == 0, sol.message
    return max(sol.fun, length.max() / ind.SPEEDS.max() + C)


def job(key):
    dataset, size, seed = key
    length = ind.lengths(dataset, size, seed)
    return dict(dataset=dataset, size=size, seed=seed, n_tasks=len(length), lb_old=ind.lower_bound(length),
                lb_min=lb_min(length), lb_lp=lb_lp(length))


def counterexample():
    """Reviewer 1: speeds 1 and 2, transfer 1, lengths 1, 1, 6; a feasible makespan is 4."""
    s, l, c = np.array([1.0, 2.0]), np.array([1.0, 1.0, 6.0]), 1.0
    old = l.sum() / s.sum() + len(l) * c / len(s)
    new = (l.sum() + len(l) * c * s.min()) / s.sum()
    return old, new


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0")
    print("counterexample old/new bound vs feasible 4:", counterexample())
    e5 = pd.read_csv(RES / "e5.csv")
    keys = list(e5[["dataset", "size", "seed"]].itertuples(index=False, name=None))
    with Pool(7) as pool:
        rows = pool.map(job, keys)
    b = pd.DataFrame(rows)
    d = e5.merge(b, on=["dataset", "size", "seed", "n_tasks"])
    gap = (d.lb_old - d.lower_bound).abs().max()
    assert gap < 1e-6, f"lengths do not reproduce E5 lower bounds ({gap})"
    for m in ["MIN_MIN", "LPT", "CSA", "CSA_OBL", "CSA_ROJ", "GA", "ROCS_LPT"]:
        assert (d[m] >= d.lb_lp - 1e-6).all(), f"{m} below LP bound"
    out = HERE.parent / "results" / "revision" / "independent_bounds.csv"
    d.drop(columns=["lb_old"]).to_csv(out, index=False)
    print("tersimpan", out)
