"""Checks of Section 4.2: the HEFT example of Topcuoglu et al. (2002) and decoder consistency.

The example has 10 tasks on three processors, and HEFT must reach a makespan of 80. The consistency
check decodes the HEFT, PEFT, and both EZ assignments of 270 E1 instances with 200 tasks under their
own priority order, 1,080 schedules in total, and requires the heuristic makespan exactly.
"""
import warnings
from multiprocessing import Pool

import numpy as np

from clustering import edge_zeroing
from model import Problem, Workflow, evaluate, fleet, generate, list_schedule, prepare
from run_workflows import RECIPES

W = np.array([[14, 16, 9], [13, 19, 18], [11, 13, 19], [13, 8, 17], [12, 13, 10], [13, 16, 9],
              [7, 15, 11], [5, 11, 14], [18, 12, 20], [21, 7, 16]], dtype=float)
E = [(0, 1, 18), (0, 2, 12), (0, 3, 9), (0, 4, 11), (0, 5, 14), (1, 7, 19), (1, 8, 16), (2, 6, 23),
     (3, 7, 27), (3, 8, 23), (4, 8, 13), (5, 7, 15), (6, 9, 17), (7, 9, 11), (8, 9, 13)]


def topcuoglu():
    n = len(W)
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    for a, b, c in E:
        children[a].append((b, c))
        parents[b].append((a, float(c)))
    rank = np.zeros(n)
    for i in reversed(range(n)):
        rank[i] = W[i].mean() + max((c + rank[j] for j, c in children[i]), default=0.0)
    order = np.array(sorted(range(n), key=lambda i: -rank[i]), dtype=np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + len(parents[i])
    parent = np.array([a for i in range(n) for a, _ in parents[i]], dtype=np.int64)
    comm = np.array([c for i in range(n) for _, c in parents[i]], dtype=float)
    wf = Workflow(W, ptr, parent, comm, order, order, np.zeros_like(W), 0.0)
    makespan, _ = list_schedule(wf, order)
    assert abs(makespan - 80) < 1e-9
    return makespan


def consistency(key):
    recipe, ccr, seed = key
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, 200, 1000 + seed)
    wf = prepare(runtime, edges, ccr, fleet(16))
    schedules = [(order,) + list_schedule(wf, order, table)
                 for order, table in ((wf.heft_order, None), (wf.peft_order, wf.oct))]
    for order in (wf.heft_order, wf.peft_order):
        makespan, x, _ = edge_zeroing(Problem(wf.exe, order, wf.ptr, wf.parent, wf.comm))
        schedules.append((order, makespan, x))
    return [abs(evaluate(Problem(wf.exe, o, wf.ptr, wf.parent, wf.comm), x)[0] - mk) for o, mk, x in schedules]


if __name__ == "__main__":
    print("Topcuoglu example, HEFT makespan:", topcuoglu())
    keys = [(r, c, s) for r in RECIPES for c in (0.1, 1.0, 5.0) for s in range(10)]
    with Pool(7) as pool:
        gaps = np.concatenate(pool.map(consistency, keys))
    print(f"Decoder consistency: {len(gaps)} schedules, largest difference {gaps.max():.3g}")
    assert gaps.max() == 0
