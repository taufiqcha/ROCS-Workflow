"""Constraint programming reference of Eqs. (9) to (11), solved with CP-SAT (OR-Tools 9.11).

Times are rounded to 0.1 s. The solver receives the best heuristic assignment as a hint.
"""
import numpy as np
from ortools.sat.python import cp_model

SCALE = 10


def solve(exe, ptr, parent, comm, hint, time_limit=120, workers=4):
    """Returns the best makespan, the solver lower bound, and the solver status."""
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
    return solver.ObjectiveValue() / SCALE, solver.BestObjectiveBound() / SCALE, solver.StatusName(status)
