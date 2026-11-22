"""Evaluation of the selection policy.

Policies, all evaluated on held-out workflow families (leave-one-family-out, nine folds):
- KEEP: never refine, return the best heuristic schedule;
- ALWAYS: run ROCS on every instance;
- TREE: the depth-three tree of E6 (selection_rule.py) fitted on the eight other families;
- GAP: refine when C_EZ/LB exceeds a threshold chosen on the eight other families by balanced accuracy.
The E6 label is kept: the best clustering-seeded method improves EZ by more than 1%. Utility is the
improvement of ROCS over the best heuristic when the policy refines, and zero otherwise. Cost is the
refinement time of ROCS. End-to-end times (constructive stage, features) are added in a8b.

Also reported: support of every leaf of the full-data tree, the trees of the GroupKFold folds used for
the 0.65 balanced accuracy, the role of the deterministic condition C_EZ/LB <= 1.01, and the transfer of
the full-data tree to the fleets of E3 (8 and 32 VMs, low and high heterogeneity), which it never saw.
Run from the repository root: python revision/selection_policy.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE.parent / "results" / "revision"
KEYS = ["workflow", "size", "ccr", "seed"]
FEATURES = ["n_tasks", "ccr", "edges_per_task", "depth", "relative_width", "parallelism", "top10_transfer_share",
            "chain_task_share", "measured_ccr", "max_out_degree", "ez_over_lb"]
H = ["HEFT", "PEFT", "EZ"]


def tree():
    return DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)


def prepare(df, feats):
    d = df.merge(feats, on=KEYS)
    d["ez_over_lb"] = d.EZ / d.lower_bound
    d["gain"] = (d.EZ - d[["ROCS", "EZ_GA", "EZ_PCSA"]].min(1)) / d.EZ * 100
    d["target"] = (d.gain > 1).astype(int)
    d["rocs_gain"] = (d[H].min(1) - d.ROCS) / d[H].min(1) * 100
    return d


def summarise(d, refine, name):
    util = np.where(refine, d.rocs_gain, 0.0)
    return dict(policy=name, refine_share=100 * refine.mean(),
                balanced_accuracy=balanced_accuracy_score(d.target, refine.astype(int)) if d.target.nunique() > 1 else np.nan,
                mean_improvement=util.mean(), retained=100 * util.mean() / d.rocs_gain.mean(),
                refinement_seconds=d.seconds_ROCS[refine].sum(),
                time_saved=100 * (1 - d.seconds_ROCS[refine].sum() / d.seconds_ROCS.sum()))


def best_threshold(train):
    grid = np.unique(np.round(train.ez_over_lb.values, 3))
    scores = [balanced_accuracy_score(train.target, (train.ez_over_lb > t).astype(int)) for t in grid]
    return grid[int(np.argmax(scores))]


def main():
    feats = pd.read_csv(RES / "e6_features.csv")
    d = prepare(pd.read_csv(RES / "e1.csv"), feats)
    pred_e6 = pd.read_csv(RES / "e6_predictions.csv")[KEYS + ["cv_prediction"]]
    d = d.merge(pred_e6, on=KEYS)
    families = sorted(d.workflow.unique())
    tree_lofo, gap_lofo = np.zeros(len(d), bool), np.zeros(len(d), bool)
    per_family, thresholds = [], {}
    for f in families:
        tr, te = d.workflow != f, d.workflow == f
        model = tree().fit(d.loc[tr, FEATURES].values, d.loc[tr, "target"].values)
        tree_lofo[te.values] = model.predict(d.loc[te, FEATURES].values) == 1
        t = best_threshold(d[tr]); thresholds[f] = t
        gap_lofo[te.values] = (d.loc[te, "ez_over_lb"] > t).values
    for f in families:
        m = (d.workflow == f).values
        sub = d[m]
        for name, ref in [("TREE", tree_lofo[m]), ("GAP", gap_lofo[m]), ("ALWAYS", np.ones(m.sum(), bool))]:
            row = summarise(sub, ref, name); row["family"] = f
            row["error_rate"] = 100 * (ref.astype(int) != sub.target.values).mean()
            per_family.append(row)
    overall = [summarise(d, np.zeros(len(d), bool), "KEEP"), summarise(d, np.ones(len(d), bool), "ALWAYS"),
               summarise(d, tree_lofo, "TREE (leave-one-family-out)"),
               summarise(d, d.cv_prediction.values == 1, "TREE (E6 GroupKFold, as in the article)"),
               summarise(d, gap_lofo, "GAP threshold (leave-one-family-out)")]
    for t in (1.01, 1.05):
        overall.append(summarise(d, (d.ez_over_lb > t).values, f"GAP fixed C_EZ/LB > {t}"))
    pd.DataFrame(overall).to_csv(OUT / "selection_policy.csv", index=False)
    pd.DataFrame(per_family).to_csv(OUT / "selection_policy_per_family.csv", index=False)

    # full-data tree, leaf support, fold trees
    full = tree().fit(d[FEATURES].values, d.target.values)
    leaf = full.apply(d[FEATURES].values)
    support = d.assign(leaf=leaf).groupby("leaf").agg(n=("target", "size"), positives=("target", "sum"),
                                                     predicted=("target", lambda s: int(full.predict(d.loc[s.index, FEATURES].values)[0])))
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(d[FEATURES], d.target, d.workflow)):
        mdl = tree().fit(d.iloc[tr][FEATURES].values, d.iloc[tr].target.values)
        folds.append((k, sorted(d.iloc[te].workflow.unique()), export_text(mdl, feature_names=FEATURES, decimals=2)))
    trivial = d.ez_over_lb <= 1.01

    # transfer of the full-data tree to the E3 fleets (structures already in e6_features)
    transfer = []
    for name in ["e3_8vm", "e3_32vm", "e3_low_heterogeneity", "e3_high_heterogeneity"]:
        e3 = prepare(pd.read_csv(RES / f"{name}.csv"), feats)
        p = full.predict(e3[FEATURES].values) == 1
        row = summarise(e3, p, "TREE full-data"); row["fleet"] = name
        transfer.append(row)
        row = summarise(e3, np.ones(len(e3), bool), "ALWAYS"); row["fleet"] = name
        transfer.append(row)
    pd.DataFrame(transfer).to_csv(OUT / "selection_policy_fleets.csv", index=False)

    with open(OUT / "selection_policy.md", "w") as f:
        f.write("# Selection policy evaluation\n\n")
        f.write(pd.DataFrame(overall).round(3).to_markdown(index=False) + "\n\n")
        f.write(f"Gap threshold chosen for each held-out family: {thresholds}\n\n")
        f.write("## Per held-out family\n\n")
        f.write(pd.DataFrame(per_family)[["family", "policy", "refine_share", "error_rate", "mean_improvement", "retained", "time_saved"]].round(2).to_markdown(index=False) + "\n\n")
        f.write("## Leaf support of the full-data tree (2,430 instances)\n\n" + export_text(full, feature_names=FEATURES, decimals=2) + "\n")
        f.write(support.to_markdown() + "\n\n")
        f.write(f"## Guaranteed condition C_EZ/LB <= 1.01\n\n{int(trivial.sum())} instances, positive labels {int(d.target[trivial].sum())}, "
                f"largest gain {d.gain[trivial].max():.3f}%. Grouped balanced accuracy of the tree outside this subset: "
                f"{balanced_accuracy_score(d.target[~trivial], d.cv_prediction[~trivial]):.3f}\n\n")
        f.write("## Trees of the GroupKFold folds\n\n")
        for k, fam, txt in folds:
            f.write(f"Fold {k}, tested on {fam}\n\n```\n{txt}```\n\n")
        f.write("## Full-data tree applied to the E3 fleets\n\n" + pd.DataFrame(transfer).round(3).to_markdown(index=False) + "\n")
    print(open(OUT / "selection_policy.md").read()[:6000])


if __name__ == "__main__":
    main()
