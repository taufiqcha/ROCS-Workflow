"""Workflow model, list-scheduling decoder, HEFT, and PEFT (Section 2 of the article).

Task i runs on VM k for runtime_i * 500 / s_k seconds. A transfer costs data / bandwidth when the
parent and the child run on different VMs and nothing otherwise. The bandwidth is set so that the
mean transfer time divided by the mean execution time equals the CCR.
"""
import random
from typing import NamedTuple

import numpy as np
from numba import njit


def fleet(m, speeds=(400, 500, 600)):
    """VM speeds in MIPS, cycling through the given speeds."""
    return np.array([speeds[k % len(speeds)] for k in range(m)], dtype=float)


class Workflow(NamedTuple):
    exe: np.ndarray
    ptr: np.ndarray
    parent: np.ndarray
    comm: np.ndarray
    heft_order: np.ndarray
    peft_order: np.ndarray
    oct: np.ndarray
    lower_bound: float


class Problem(NamedTuple):
    """Arrays shared by every method: the parents of task i are parent[ptr[i]:ptr[i + 1]]."""
    exe: np.ndarray
    order: np.ndarray
    ptr: np.ndarray
    parent: np.ndarray
    comm: np.ndarray


def generate(recipe, n, seed):
    """Trace-based workflow from a WfCommons 1.5 recipe as run times and (parent, child, bytes) edges."""
    from wfcommons import WorkflowGenerator
    from wfcommons.wfchef import recipes
    random.seed(seed)
    np.random.seed(seed)
    wf = WorkflowGenerator(getattr(recipes, recipe + "Recipe").from_num_tasks(n)).build_workflow()
    names = list(wf.nodes)
    index = {t: i for i, t in enumerate(names)}
    tasks = [wf.nodes[t]["task"] for t in names]
    runtime = np.array([max(t.runtime, 1e-3) for t in tasks])
    outputs = [{f.file_id: f.size for f in t.output_files} for t in tasks]
    edges = []
    for a, b in wf.edges:
        ia, ib = index[a], index[b]
        edges.append((ia, ib, float(sum(outputs[ia].get(f.file_id, 0) for f in tasks[ib].input_files))))
    return runtime, edges


def topological(n, parents, children):
    indegree = np.array([len(parents[i]) for i in range(n)])
    stack = [i for i in range(n) if indegree[i] == 0]
    order = []
    while stack:
        i = stack.pop()
        order.append(i)
        for j, _ in children[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                stack.append(j)
    return order


def prepare(runtime, edges, ccr, speeds):
    """Execution and transfer times, the HEFT and PEFT orders, and the lower bound of an instance."""
    n, m = len(runtime), len(speeds)
    exe = runtime[:, None] * 500.0 / speeds[None, :]
    data = np.array([d for _, _, d in edges])
    if data.sum() == 0:
        data = np.ones_like(data)
    bandwidth = data.mean() / (ccr * exe.mean())
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for (a, b, _), d in zip(edges, data):
        parents[b].append((a, d / bandwidth))
        children[a].append((b, d / bandwidth))
    topo = topological(n, parents, children)

    mean_exe = exe.mean(1)
    rank = np.zeros(n)
    for i in reversed(topo):
        rank[i] = mean_exe[i] + max((c + rank[j] for j, c in children[i]), default=0.0)
    heft_order = np.array(sorted(range(n), key=lambda i: -rank[i]), dtype=np.int64)

    ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + len(parents[i])
    parent = np.array([a for i in range(n) for a, _ in parents[i]], dtype=np.int64)
    comm = np.array([c for i in range(n) for _, c in parents[i]], dtype=float)

    path = np.zeros(n)
    for i in topo:
        path[i] = exe[i].min() + max((path[a] for a, _ in parents[i]), default=0.0)
    lower_bound = max(path.max(), (runtime * 500.0).sum() / speeds.sum())

    oct_table = np.zeros((n, m))
    for i in reversed(topo):
        for k in range(m):
            oct_table[i, k] = max((min(oct_table[j, w] + exe[j, w] + (c if w != k else 0.0) for w in range(m))
                                   for j, c in children[i]), default=0.0)
    # PEFT takes the ready task with the largest mean OCT; a global sort by mean OCT is not topological.
    rank_oct = oct_table.mean(1)
    waiting = np.array([len(parents[i]) for i in range(n)])
    ready = [i for i in range(n) if waiting[i] == 0]
    peft_order = []
    while ready:
        i = max(ready, key=lambda t: rank_oct[t])
        ready.remove(i)
        peft_order.append(i)
        for j, _ in children[i]:
            waiting[j] -= 1
            if waiting[j] == 0:
                ready.append(j)
    peft_order = np.array(peft_order, dtype=np.int64)

    for order in (heft_order, peft_order):
        position = np.empty(n, dtype=np.int64)
        position[order] = np.arange(n)
        assert all(position[a] < position[b] for b in range(n) for a, _ in parents[b]), "order is not topological"
    return Workflow(exe, ptr, parent, comm, heft_order, peft_order, oct_table, lower_bound)


@njit(cache=True)
def insertion_slot(start, finish, count, est, duration):
    """Earliest start not before est in an idle gap of one VM that fits the duration."""
    t = est
    for s in range(count):
        if t + duration <= start[s]:
            return t, s
        if finish[s] > t:
            t = finish[s]
    return t, count


@njit(cache=True)
def decode(exe, order, ptr, parent, comm, x, choose_vm, oct_table):
    """List scheduler of Eq. (3). With choose_vm the VM minimizing finish time plus OCT is chosen and
    written to x; otherwise every task runs on VM x[i]. Returns the makespan and the busy time per VM."""
    n, m = exe.shape
    start = np.zeros((m, n))
    finish = np.zeros((m, n))
    count = np.zeros(m, dtype=np.int64)
    done = np.zeros(n)
    for q in range(n):
        i = order[q]
        k0, k1 = (0, m) if choose_vm else (x[i], x[i] + 1)
        best_f, best_k, best_s, best_pos = 1e300, -1, 0.0, 0
        for k in range(k0, k1):
            est = 0.0
            for e in range(ptr[i], ptr[i + 1]):
                a = parent[e]
                t = done[a] + (comm[e] if x[a] != k else 0.0)
                if t > est:
                    est = t
            s, pos = insertion_slot(start[k], finish[k], count[k], est, exe[i, k])
            if s + exe[i, k] + oct_table[i, k] < best_f:
                best_f, best_k, best_s, best_pos = s + exe[i, k] + oct_table[i, k], k, s, pos
        k = best_k
        for s in range(count[k], best_pos, -1):
            start[k, s] = start[k, s - 1]
            finish[k, s] = finish[k, s - 1]
        start[k, best_pos] = best_s
        finish[k, best_pos] = best_s + exe[i, k]
        count[k] += 1
        done[i] = best_s + exe[i, k]
        x[i] = k
    busy = np.zeros(m)
    for k in range(m):
        for s in range(count[k]):
            busy[k] += finish[k, s] - start[k, s]
    return done.max(), busy


def list_schedule(wf, order, oct_table=None):
    """HEFT with the upward-rank order, or PEFT with the PEFT order and its optimistic cost table."""
    x = np.full(wf.exe.shape[0], -1, dtype=np.int64)
    if oct_table is None:
        oct_table = np.zeros_like(wf.exe)
    makespan, _ = decode(wf.exe, order, wf.ptr, wf.parent, wf.comm, x, True, oct_table)
    return makespan, x


def evaluate(problem, x):
    """Makespan and busy time per VM of assignment x under the shared priority order."""
    return decode(problem.exe, problem.order, problem.ptr, problem.parent, problem.comm, x.copy(), False,
                  np.zeros_like(problem.exe))


def heuristic_seeds(wf):
    """HEFT, PEFT, and EZ with both orders. The best schedule fixes the priority order shared by all
    methods, and the three best schedules seed the clustering-seeded methods."""
    from clustering import edge_zeroing
    mk_heft, x_heft = list_schedule(wf, wf.heft_order)
    mk_peft, x_peft = list_schedule(wf, wf.peft_order, wf.oct)
    candidates = [(mk_heft, x_heft, wf.heft_order), (mk_peft, x_peft, wf.peft_order)]
    ez = []
    for order in (wf.heft_order, wf.peft_order):
        mk_ez, x_ez, _ = edge_zeroing(Problem(wf.exe, order, wf.ptr, wf.parent, wf.comm))
        ez.append((mk_ez, x_ez, order))
    candidates = sorted(candidates + ez, key=lambda c: c[0])
    problem = Problem(wf.exe, candidates[0][2], wf.ptr, wf.parent, wf.comm)
    assert abs(evaluate(problem, candidates[0][1])[0] - candidates[0][0]) < 1e-6 * candidates[0][0]
    makespans = {"HEFT": mk_heft, "PEFT": mk_peft, "EZ": min(c[0] for c in ez)}
    return problem, makespans, [c[1] for c in candidates[:3]], [x_heft, x_peft]
