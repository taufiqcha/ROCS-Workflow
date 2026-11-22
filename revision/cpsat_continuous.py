"""CP-SAT schedules evaluated on the continuous time scale of the decoders.

The model of exact.py rounds durations and transfer times to 0.1 s. This script solves the same model
(same hint, 120 s, 4 workers) and keeps the solver schedule. The schedule is then replayed with the
exact durations: tasks are taken in order of their solver start times, each task stays on its solver
VM and after its predecessor on that VM, and starts when its data have arrived and that VM is free.
The replayed makespan is a feasible continuous schedule, so all methods are compared on one scale.
The solver bound applies to the rounded instance only.

Run alone from the repository root: python revision/cpsat_continuous.py
"""
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
os.chdir(CODE)
from exact import SCALE  # noqa: E402
from run_e4 import SIZES, instance  # noqa: E402


def solve_keep(exe, ptr, parent, comm, hint, time_limit=120, workers=4):
    """exact.solve with the solution kept. Model lines are identical to exact.py."""
    n, m = exe.shape
    d = np.maximum(1, np.round(exe * SCALE)).astype(int)
    c = np.round(comm * SCALE).astype(int)
    horizon = int(d.max(1).sum() + c.sum())
    model = cp_model.CpModel()
    y = [[model.NewBoolVar(f"y{i}_{k}") for k in range(m)] for i in range(n)]
    start = [model.NewIntVar(0, horizon, f"s{i}") for i in range(n)]
    end = [model.NewIntVar(0, horizon, f"e{i}") for i in range(n)]
    intervals = [[] for _ in range(m)]
    for i in range(n):
        model.AddExactlyOne(y[i])
        model.Add(end[i] == start[i] + sum(int(d[i, k]) * y[i][k] for k in range(m)))
        for k in range(m):
            intervals[k].append(model.NewOptionalFixedSizeIntervalVar(start[i], int(d[i, k]), y[i][k], f"iv{i}_{k}"))
    for k in range(m):
        model.AddNoOverlap(intervals[k])
    for b in range(n):
        for q in range(ptr[b], ptr[b + 1]):
            a = int(parent[q])
            same = model.NewBoolVar(f"z{a}_{b}")
            both = [model.NewBoolVar(f"w{a}_{b}_{k}") for k in range(m)]
            for k in range(m):
                model.AddImplication(both[k], y[a][k])
                model.AddImplication(both[k], y[b][k])
                model.AddBoolOr([y[a][k].Not(), y[b][k].Not(), both[k]])
            model.Add(sum(both) == same)
            model.Add(start[b] >= end[a] + int(c[q])).OnlyEnforceIf(same.Not())
            model.Add(start[b] >= end[a]).OnlyEnforceIf(same)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, end)
    model.Minimize(makespan)
    for i in range(n):
        for k in range(m):
            model.AddHint(y[i][k], int(hint[i] == k))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    x = np.array([next(k for k in range(m) if solver.Value(y[i][k])) for i in range(n)])
    s = np.array([solver.Value(start[i]) for i in range(n)])
    return solver.ObjectiveValue() / SCALE, solver.BestObjectiveBound() / SCALE, solver.StatusName(status), x, s


def replay(exe, ptr, parent, comm, x, s):
    n, m = exe.shape
    free = np.zeros(m)
    done = np.zeros(n)
    for i in sorted(range(n), key=lambda t: (s[t], t)):
        k = x[i]
        est = free[k]
        for e in range(ptr[i], ptr[i + 1]):
            a = parent[e]
            est = max(est, done[a] + (comm[e] if x[a] != k else 0.0))
        done[i] = est + exe[i, k]
        free[k] = done[i]
    return done.max()


def main():
    warnings.filterwarnings("ignore")
    out = HERE.parent / "results" / "revision" / "cpsat_continuous.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow", "ccr", "seed", "n_tasks", "status", "cpsat_rounded", "cpsat_bound_rounded",
                    "cpsat_continuous", "seconds"])
        for recipe in SIZES:
            for ccr in (1.0, 5.0):
                for seed in range(3):
                    wf, _, (problem, _, ez_seeds, _) = instance(recipe, ccr, seed)
                    t = time.time()
                    best, bound, status, x, s = solve_keep(wf.exe, wf.ptr, wf.parent, wf.comm, ez_seeds[0])
                    cont = replay(wf.exe, wf.ptr, wf.parent, wf.comm, x, s)
                    w.writerow([recipe, ccr, seed, wf.exe.shape[0], status, best, bound, cont, time.time() - t])
                    f.flush()
                    print(recipe, ccr, seed, status, best, round(cont, 2), flush=True)


if __name__ == "__main__":
    main()
