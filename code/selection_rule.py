"""E6: selection rule of Section 3.5, a depth-three decision tree on instance features.

The target is whether the best clustering-seeded method (ROCS, EZ+GA, or EZ+PCSA) improves EZ by
more than 1%. Five-fold cross-validation is grouped by workflow, so the tested application is never
seen in training. Writes the features, the cross-validated predictions, and the tree fitted on all
E1 instances.
"""
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from model import fleet, generate, prepare

KEYS = ["workflow", "size", "ccr", "seed"]
FEATURES = ["n_tasks", "ccr", "edges_per_task", "depth", "relative_width", "parallelism", "top10_transfer_share",
            "chain_task_share", "measured_ccr", "max_out_degree", "ez_over_lb"]


def features(key):
    recipe, size, ccr, seed = key
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, size, 1000 + seed)
    wf = prepare(runtime, edges, ccr, fleet(16))
    n = len(runtime)
    parents = [list(wf.parent[wf.ptr[i]:wf.ptr[i + 1]]) for i in range(n)]
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for a in parents[i]:
            out_degree[a] += 1
    level = np.zeros(n, dtype=int)
    path = np.zeros(n)
    for i in wf.heft_order:
        level[i] = 1 + max((level[a] for a in parents[i]), default=-1)
        path[i] = wf.exe[i].min() + max((path[a] for a in parents[i]), default=0.0)
    transfers = np.sort(wf.comm)[::-1]
    top = max(1, len(transfers) // 10)
    chained = sum(1 for i in range(n) if len(parents[i]) == 1 and out_degree[parents[i][0]] == 1)
    return dict(workflow=recipe, size=size, ccr=ccr, seed=seed,
                edges_per_task=len(wf.comm) / n, depth=level.max() + 1,
                relative_width=np.bincount(level).max() / n,
                parallelism=wf.exe.min(1).sum() / path.max(),
                top10_transfer_share=transfers[:top].sum() / max(transfers.sum(), 1e-12),
                chain_task_share=chained / n,
                measured_ccr=wf.comm.mean() / wf.exe.mean() if len(wf.comm) else 0.0,
                max_out_degree=out_degree.max() / n)


def tree():
    return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e1", default="../results/e1.csv")
    p.add_argument("--workers", type=int, default=7)
    a = p.parse_args()
    e1 = pd.read_csv(a.e1)
    with Pool(a.workers) as pool:
        F = pd.DataFrame(pool.map(features, list(e1[KEYS].drop_duplicates().itertuples(index=False, name=None))))
    F.to_csv("../results/e6_features.csv", index=False)
    d = e1.merge(F, on=KEYS)
    d["gain"] = (d.EZ - d[["ROCS", "EZ_GA", "EZ_PCSA"]].min(1)) / d.EZ * 100
    d["ez_over_lb"] = d.EZ / d.lower_bound
    d["target"] = (d.gain > 1).astype(int)
    X, y, groups = d[FEATURES].values, d.target.values, d.workflow.values
    prediction = np.zeros_like(y)
    for train, test in GroupKFold(5).split(X, y, groups):
        prediction[test] = tree().fit(X[train], y[train]).predict(X[test])
    d["cv_prediction"] = prediction
    d[KEYS + ["target", "cv_prediction", "gain"]].to_csv("../results/e6_predictions.csv", index=False)
    model = tree().fit(X, y)
    importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    report = [f"Balanced accuracy, grouped cross-validation: {balanced_accuracy_score(y, prediction):.3f}", "",
              export_text(model, feature_names=FEATURES, decimals=2), "Feature importance"]
    report += [f"  {name}: {value:.3f}" for name, value in importance if value > 0]
    with open("../results/e6_tree.txt", "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
