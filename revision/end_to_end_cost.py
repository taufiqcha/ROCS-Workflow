"""End-to-end cost of the selection policies.

Feature computation (selection_rule.features without instance generation) is timed on one seed per
workflow, size, and CCR (81 E1 instances, 4 processes). The constructive stage uses seconds_EZ of E1
(HEFT, PEFT, and EZ with both orders) and refinement uses seconds_ROCS of E1.
Run from the repository root: python revision/end_to_end_cost.py
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))
os.chdir(CODE)
import selection_rule as sr  # noqa: E402
from model import fleet, generate, prepare  # noqa: E402

RECIPES = ["Blast", "Bwa", "Cycles", "Epigenomics", "Genome", "Montage", "Seismology", "Soykb", "Srasearch"]


def timed(job):
    recipe, size, ccr = job
    warnings.filterwarnings("ignore")
    runtime, edges = generate(recipe, size, 1000)
    wf = prepare(runtime, edges, ccr, fleet(16))
    real_generate, real_prepare = sr.generate, sr.prepare
    sr.generate, sr.prepare = (lambda *a: (runtime, edges)), (lambda *a: wf)
    t = time.perf_counter()
    sr.features((recipe, size, ccr, 0))
    seconds = time.perf_counter() - t
    sr.generate, sr.prepare = real_generate, real_prepare
    return dict(workflow=recipe, size=size, ccr=ccr, feature_seconds=seconds)


def measure():
    jobs = [(r, n, c) for r in RECIPES for n in (200, 500, 1000) for c in (0.1, 1.0, 5.0)]
    with Pool(4) as pool:
        f = pd.DataFrame(pool.map(timed, jobs))
    f.to_csv(HERE.parent / "results" / "revision" / "feature_time.csv", index=False)
    print(f.groupby("size").feature_seconds.describe().round(4))


def end_to_end():
    """End-to-end time of the policies of selection_policy.py, from feature_time.csv and E1."""
    sys.path.insert(0, str(HERE))
    import selection_policy as A
    feats = pd.read_csv(A.RES / "e6_features.csv")
    d = A.prepare(pd.read_csv(A.RES / "e1.csv"), feats).merge(
        pd.read_csv(A.RES / "e6_predictions.csv")[A.KEYS + ["cv_prediction"]], on=A.KEYS)
    ft = pd.read_csv(HERE.parent / "results" / "revision" / "feature_time.csv").groupby("size").feature_seconds.mean()
    d["feat_s"] = d["size"].map(ft)
    tree_lofo, gap_lofo = np.zeros(len(d), bool), np.zeros(len(d), bool)
    for f in sorted(d.workflow.unique()):
        tr, te = d.workflow != f, d.workflow == f
        tree_lofo[te.values] = A.tree().fit(d.loc[tr, A.FEATURES].values, d.loc[tr, "target"].values).predict(d.loc[te, A.FEATURES].values) == 1
        gap_lofo[te.values] = (d.loc[te, "ez_over_lb"] > A.best_threshold(d[tr])).values
    policies = {"KEEP": np.zeros(len(d), bool), "ALWAYS": np.ones(len(d), bool), "TREE (GroupKFold E6)": d.cv_prediction.values == 1,
                "TREE (leave-one-family-out)": tree_lofo, "GAP (leave-one-family-out)": gap_lofo, "GAP C_EZ/LB > 1.05": (d.ez_over_lb > 1.05).values}
    rows = []
    always = d.seconds_EZ.sum() + d.seconds_ROCS.sum()
    for name, r in policies.items():
        total = d.seconds_EZ.sum() + (d.feat_s.sum() if name.startswith("TREE") else 0) + d.seconds_ROCS[r].sum()
        util = np.where(r, d.rocs_gain, 0).mean()
        rows.append(dict(policy=name, mean_improvement=util, retained=100 * util / d.rocs_gain.mean(),
                         end_to_end_s=total, relative_to_always=100 * total / always))
    pd.DataFrame(rows).to_csv(HERE.parent / "results" / "revision" / "end_to_end_cost.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    if "--skip-timing" not in sys.argv:
        measure()
    end_to_end()
