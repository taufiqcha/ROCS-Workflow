"""Edge-zeroing clustering (EZ) of Sarkar (1989) for heterogeneous VMs (Section 2.4).

Edges are visited in decreasing order of transfer time, and the clusters of both endpoints are merged
whenever the makespan does not increase. All tasks of a cluster run on one VM, which is chosen by
earliest finish time when the first task of the cluster is scheduled.
"""
import numpy as np
from numba import njit

from model import insertion_slot


@njit(cache=True)
def decode_clusters(exe, order, ptr, parent, comm, cluster, x):
    n, m = exe.shape
    start = np.zeros((m, n))
    finish = np.zeros((m, n))
    count = np.zeros(m, dtype=np.int64)
    done = np.zeros(n)
    cluster_vm = np.full(n, -1, dtype=np.int64)
    for q in range(n):
        i = order[q]
        c = cluster[i]
        k0, k1 = (cluster_vm[c], cluster_vm[c] + 1) if cluster_vm[c] >= 0 else (0, m)
        best_f, best_k, best_s, best_pos = 1e300, -1, 0.0, 0
        for k in range(k0, k1):
            est = 0.0
            for e in range(ptr[i], ptr[i + 1]):
                a = parent[e]
                t = done[a] + (comm[e] if x[a] != k else 0.0)
                if t > est:
                    est = t
            s, pos = insertion_slot(start[k], finish[k], count[k], est, exe[i, k])
            if s + exe[i, k] < best_f:
                best_f, best_k, best_s, best_pos = s + exe[i, k], k, s, pos
        k = best_k
        cluster_vm[c] = k
        for s in range(count[k], best_pos, -1):
            start[k, s] = start[k, s - 1]
            finish[k, s] = finish[k, s - 1]
        start[k, best_pos] = best_s
        finish[k, best_pos] = best_f
        count[k] += 1
        done[i] = best_f
        x[i] = k
    return done.max()


def edge_zeroing(problem):
    """Returns the makespan, the assignment, and the number of schedules evaluated."""
    exe, order, ptr, parent, comm = problem
    n = exe.shape[0]
    child = np.repeat(np.arange(n), np.diff(ptr))
    cluster = np.arange(n)

    def root(i):
        while cluster[i] != i:
            cluster[i] = cluster[cluster[i]]
            i = cluster[i]
        return i

    def schedule():
        labels = np.array([root(i) for i in range(n)], dtype=np.int64)
        x = np.full(n, -1, dtype=np.int64)
        return decode_clusters(exe, order, ptr, parent, comm, labels, x), x

    makespan, x_best = schedule()
    evaluations = 1
    for e in np.argsort(-comm):
        a, b = root(parent[e]), root(child[e])
        if a == b:
            continue
        saved = cluster.copy()
        cluster[b] = a
        candidate, x = schedule()
        evaluations += 1
        if candidate <= makespan:
            makespan, x_best = candidate, x
        else:
            cluster[:] = saved
    return makespan, x_best, evaluations
