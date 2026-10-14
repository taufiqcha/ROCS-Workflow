"""Recomputes every value reported in Section 5 from the archived results and writes summary.md.

Improvement is the relative makespan reduction in percent with respect to the best heuristic on the
same instance. Pairwise tests are paired Wilcoxon signed-rank tests, and the paired A12 is the share
of instances on which the first method is shorter, with ties counted as one half.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon
from sklearn.metrics import balanced_accuracy_score

R = Path(__file__).resolve().parents[1] / "results"
HEURISTICS = ["HEFT", "PEFT", "EZ"]
E1_METHODS = HEURISTICS + ["PSO", "CSA", "CSA_OBL", "GA", "PCSA", "ROCS", "EZ_GA", "EZ_PCSA"]
lines = []


def out(text=""):
    lines.append(text)


def gain(reference, method):
    return (reference - method) / reference * 100


def p_value(a, b):
    return 1.0 if np.allclose(a, b) else wilcoxon(a, b, zero_method="zsplit").pvalue


def holm(p):
    p = np.asarray(p)
    adjusted, running = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - r) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def win_tie_loss(a, b, tol=1e-9):
    wins, losses = int((a < b - tol).sum()), int((a > b + tol).sum())
    return wins, len(a) - wins - losses, losses


def independent_tasks():
    d = pd.read_csv(R / "e5.csv")
    methods = ["MIN_MIN", "LPT", "CSA", "CSA_OBL", "CSA_ROJ", "GA", "ROCS_LPT"]
    ratio = d.copy()
    ratio[methods] = d[methods].div(d.lower_bound, axis=0)
    median = ratio.groupby(["dataset", "size"])[methods].median()
    out("## 5.1 Independent tasks (E5)")
    out()
    out("Median makespan relative to the lower bound (Table 3 and Fig. 2)")
    out()
    out("| Dataset | Tasks | " + " | ".join(methods) + " |")
    out("|---|---|" + "---|" * len(methods))
    for (dataset, size), row in median.iterrows():
        out(f"| {dataset} | {size} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    out()
    out(f"- LPT within 1.5% of the bound: {int((median.LPT <= 1.015).sum())} of {len(median)} configurations")
    out(f"- ROCS from LPT shorter than LPT: {int((d.ROCS_LPT < d.LPT - 1e-9).sum())} of {len(d)} runs")
    out(f"- GA from LPT tied with LPT: {win_tie_loss(d.GA, d.LPT)[1]} of {len(d)} runs")
    obl = [p_value(g.CSA_OBL.values, g.CSA.values) for _, g in d.groupby("dataset")]
    roj = [p_value(g.CSA_ROJ.values, g.CSA.values) for _, g in d.groupby("dataset")]
    out(f"- CSA-OBL against CSA, p per dataset: {min(obl):.2f} to {max(obl):.2f}")
    out(f"- CSA-ROJ against CSA, largest p per dataset: {max(roj):.2g}")
    out(f"- CSA-ROJ shorter than CSA: {int((d.CSA_ROJ < d.CSA).sum())} of {len(d)} runs")
    large = median[median.index.get_level_values("size") >= 3000]
    out(f"- Median gap to the bound from 3,000 tasks: CSA-ROJ {100 * (large.CSA_ROJ.min() - 1):.1f}% to "
        f"{100 * (large.CSA_ROJ.max() - 1):.1f}%, CSA {100 * (large.CSA.min() - 1):.1f}% to {100 * (large.CSA.max() - 1):.1f}%")
    out()


def workflow_benchmark():
    d = pd.read_csv(R / "e1.csv")
    best = d[HEURISTICS].min(1)
    ranks = d[E1_METHODS].rank(axis=1).mean().sort_values()
    k, n = len(E1_METHODS), len(d)
    stat = friedmanchisquare(*[d[m] for m in E1_METHODS])
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    out("## 5.2 Workflow benchmark (E1)")
    out()
    out(f"Friedman chi-square {stat.statistic:.0f}, p {stat.pvalue:.2g}, Nemenyi critical difference {cd:.3f} (Fig. 3)")
    out()
    out("| Method | Average rank |")
    out("|---|---|")
    for m, v in ranks.items():
        out(f"| {m} | {v:.2f} |")
    out()
    d["ez_over_list"] = gain(d[["HEFT", "PEFT"]].min(1), d.EZ)
    d["rocs"] = gain(best, d.ROCS)
    heat_a = d.pivot_table(index="workflow", columns="ccr", values="ez_over_list")
    me = heat_a.loc[["Epigenomics", "Montage"]]
    out(f"- EZ over HEFT and PEFT on Montage and Epigenomics (Fig. 4a): CCR 1 {me[1.0].min():.1f}% to "
        f"{me[1.0].max():.1f}%, CCR 5 {me[5.0].min():.1f}% to {me[5.0].max():.1f}%")
    by_ccr = d.groupby("ccr").rocs.mean()
    by_workflow = d.groupby("workflow").rocs.mean()
    out(f"- Mean ROCS improvement (Fig. 4b): CCR 0.1 {by_ccr[0.1]:.2f}%, CCR 5 {by_ccr[5.0]:.2f}%, "
        f"largest workflow mean {by_workflow.max():.2f}% ({by_workflow.idxmax()})")
    out(f"- ROCS more than 1% better than the best heuristic at CCR 5: {100 * (d.rocs[d.ccr == 5.0] > 1).mean():.1f}% of runs")
    out()
    out("Pairwise comparison of ROCS (Table 4)")
    out()
    out("| Opponent | Wins | Ties | Losses | p Holm | Paired A12 |")
    out("|---|---|---|---|---|---|")
    opponents = ["EZ", "EZ_GA", "EZ_PCSA"]
    adjusted = holm([p_value(d.ROCS.values, d[m].values) for m in opponents])
    for m, p in zip(opponents, adjusted):
        w, t, l = win_tie_loss(d.ROCS, d[m])
        out(f"| {m} | {w} | {t} | {l} | {p:.2g} | {(w + 0.5 * t) / n:.3f} |")
    out()
    seconds = d.groupby("size")[["seconds_ROCS", "seconds_EZ_GA"]].mean()
    for size, row in seconds.iterrows():
        out(f"- Mean run time, {size} tasks: ROCS {row.seconds_ROCS:.2f} s, EZ+GA {row.seconds_EZ_GA:.2f} s")
    sub = d[d.workflow.isin(["Montage", "Epigenomics"]) & (d.ccr >= 1)]
    out("- Mean makespan reduction of ROCS on Montage and Epigenomics at CCR 1 and 5: "
        + ", ".join(f"{m} {gain(sub[m], sub.ROCS).mean():.1f}%" for m in ["PEFT", "HEFT", "CSA_OBL", "GA"]))
    out()


def sensitivity():
    e1 = pd.read_csv(R / "e1.csv")
    subset = e1[(e1["size"] == 500) & (e1.seed < 10) & e1.ccr.isin([1.0, 5.0])]
    budgets = {"20n": subset[subset.workflow.isin(["Epigenomics", "Montage", "Genome", "Cycles"])],
               "100n": pd.read_csv(R / "e2_100n.csv"), "500n": pd.read_csv(R / "e2_500n.csv")}
    out("## 5.3 Sensitivity (E2 and E3)")
    out()
    out("Mean improvement over the best heuristic by budget (Fig. 5)")
    out()
    shown = ["ROCS", "EZ_GA", "EZ_PCSA", "GA", "CSA_OBL"]
    out("| Budget | " + " | ".join(shown) + " |")
    out("|---|" + "---|" * len(shown))
    for label, d in budgets.items():
        best = d[HEURISTICS].min(1)
        out(f"| {label} | " + " | ".join(f"{gain(best, d[m]).mean():.1f}" for m in shown) + " |")
    out()
    fleets = {"16 VMs, 400/500/600": subset[subset.workflow.isin(["Epigenomics", "Montage", "Genome", "Cycles", "Seismology"])]}
    for name, label in [("8vm", "8 VMs"), ("32vm", "32 VMs"), ("low_heterogeneity", "16 VMs, 450/500/550"),
                        ("high_heterogeneity", "16 VMs, 250/500/1000")]:
        fleets[label] = pd.read_csv(R / f"e3_{name}.csv")
    out("| Fleet | EZ over HEFT and PEFT | ROCS over best heuristic |")
    out("|---|---|---|")
    for label, d in fleets.items():
        out(f"| {label} | {gain(d[['HEFT', 'PEFT']].min(1), d.EZ).mean():.1f} | {gain(d[HEURISTICS].min(1), d.ROCS).mean():.1f} |")
    out()


def optimality_and_dual():
    d = pd.read_csv(R / "e4.csv")
    methods = ["HEFT", "EZ", "GA", "CSA_OBL", "EZ_GA", "EZ_PCSA", "ROCS", "DCSA_20", "DCSA_100", "DCSA_500", "DCSA_2000"]
    best = d[["cpsat_best"] + methods].min(1)
    proven = d[d.cpsat_status == "OPTIMAL"]
    out("## 5.4 Optimality gap (E4) and dual encoding (E7)")
    out()
    out(f"- CP-SAT proved optimality on {len(proven)} of {len(d)} instances and returned the best known "
        f"solution on {int((d.cpsat_best <= best + 1e-9).sum())}")
    to_opt = lambda m: gain(proven.cpsat_best, proven[m]).abs()
    out("- Median gap to the proven optimum: " + ", ".join(f"{m} {to_opt(m).median():.1f}%" for m in ["HEFT", "EZ", "ROCS"]))
    out("- Largest gap to the proven optimum: " + ", ".join(
        f"{m} {to_opt(m).max():.1f}%" for m in ["EZ", "GA", "CSA_OBL", "EZ_GA", "EZ_PCSA", "ROCS"]))
    out("- Median gap of DCSA to the best known solution (Fig. 6): " + ", ".join(
        f"{b} {(-gain(best, d[f'DCSA_{b}'])).median():.1f}%" for b in [20, 100, 500, 2000]))
    out(f"- DCSA at 2000n shorter than EZ+PCSA on {int((d.DCSA_2000 < d.EZ_PCSA - 1e-6).sum())} instances "
        f"and shorter than CP-SAT on {int((d.DCSA_2000 < d.cpsat_best - 1e-6).sum())}")
    out()
    e7 = pd.read_csv(R / "e7.csv")
    out("Dual encoding at 100n, improvement over EZ (Table 5)")
    out()
    out("| Method | Mean (%) | Median (%) | DCSA wins | DCSA losses | p |")
    out("|---|---|---|---|---|---|")
    for m in ["GA_DUAL", "EZ_GA", "DCSA", "DCSA_NO_KEYS", "DCSA_NO_OBL", "ROCS", "EZ_PCSA"]:
        g = gain(e7.EZ, e7[m])
        if m == "DCSA":
            out(f"| {m} | {g.mean():.2f} | {g.median():.2f} | | | |")
            continue
        w, _, l = win_tie_loss(e7.DCSA, e7[m], 1e-6)
        out(f"| {m} | {g.mean():.2f} | {g.median():.2f} | {w} | {l} | {p_value(e7.DCSA.values, e7[m].values):.3g} |")
    out()


def selection_rule():
    e1 = pd.read_csv(R / "e1.csv")
    d = e1.merge(pd.read_csv(R / "e6_predictions.csv"), on=["workflow", "size", "ccr", "seed"])
    selected = d.cv_prediction == 1
    rocs = gain(d[HEURISTICS].min(1), d.ROCS)
    framework = np.where(selected, rocs, 0.0).mean()
    out("## 5.5 Selection rule (E6)")
    out()
    out(f"- Balanced accuracy with grouped cross-validation: {balanced_accuracy_score(d.target, d.cv_prediction):.2f}")
    out(f"- ROCS invoked on {100 * selected.mean():.1f}% of the instances, refinement time saved "
        f"{100 * (1 - d.seconds_ROCS[selected].sum() / d.seconds_ROCS.sum()):.1f}%")
    out(f"- Mean improvement over the best heuristic: framework {framework:.2f}%, ROCS on every instance {rocs.mean():.2f}%, "
        f"retained {100 * framework / rocs.mean():.0f}%")
    out(f"- Runs with more than 5% improvement selected: {int((selected & (rocs > 5)).sum())} of {int((rocs > 5).sum())}")
    out()


if __name__ == "__main__":
    out("# Values reported in Section 5")
    out()
    for section in (independent_tasks, workflow_benchmark, sensitivity, optimality_and_dual, selection_rule):
        section()
    (R / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
