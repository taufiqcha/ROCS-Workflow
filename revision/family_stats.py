"""Family-level effects and uncertainty for the E1 comparisons.

The 2,430 runs come from nine workflow families, so runs of one family are not independent. This
script reports, for ROCS against each opponent:
- per-family wins, ties, losses, paired A12, and mean improvement;
- a two-level bootstrap (families resampled with replacement, then runs within each drawn family)
  for the pooled paired A12 and the mean improvement, 10,000 resamples, seed 0;
- a family-level sensitivity analysis: the Wilcoxon signed-rank test on the nine family means, and the
  Friedman test with the Nemenyi critical difference on the nine family mean ranks.
Run from the repository root: python revision/family_stats.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE.parent / "results" / "revision"
METHODS = ["HEFT", "PEFT", "EZ", "PSO", "CSA", "CSA_OBL", "GA", "PCSA", "ROCS", "EZ_GA", "EZ_PCSA"]
OPPONENTS = ["EZ", "EZ_GA", "EZ_PCSA", "GA", "PEFT"]


TOL = 1e-9  # same tie tolerance as win_tie_loss in analysis.py


def a12(a, b):
    """Share of runs where a is shorter than b, ties (within TOL) counted as one half."""
    win, loss = a < b - TOL, a > b + TOL
    return (win + 0.5 * (~win & ~loss)).mean()


def main():
    d = pd.read_csv(RES / "e1.csv")
    rng = np.random.default_rng(0)
    families = d.workflow.unique()
    groups = {f: d[d.workflow == f] for f in families}
    rows = []
    for opp in OPPONENTS:
        imp = (d[opp] - d.ROCS) / d[opp] * 100
        for f in families:
            g = groups[f]
            w, l = (g.ROCS < g[opp] - TOL).sum(), (g.ROCS > g[opp] + TOL).sum(); t = len(g) - w - l
            rows.append(dict(opponent=opp, family=f, wins=w, ties=t, losses=l, a12=a12(g.ROCS.values, g[opp].values),
                             mean_improvement=((g[opp] - g.ROCS) / g[opp] * 100).mean()))
        boot_a, boot_m = [], []
        for _ in range(10000):
            pick = rng.choice(families, len(families))
            parts = [groups[f].iloc[rng.integers(0, len(groups[f]), len(groups[f]))] for f in pick]
            s = pd.concat(parts)
            boot_a.append(a12(s.ROCS.values, s[opp].values))
            boot_m.append(((s[opp] - s.ROCS) / s[opp] * 100).mean())
        fam_mean = pd.DataFrame([r for r in rows if r["opponent"] == opp])
        p_fam = wilcoxon(fam_mean.mean_improvement, zero_method="zsplit").pvalue if (fam_mean.mean_improvement != 0).any() else 1.0
        W, L = (d.ROCS < d[opp] - TOL).sum(), (d.ROCS > d[opp] + TOL).sum()
        rows.append(dict(opponent=opp, family="ALL", wins=W, ties=len(d) - W - L,
                         losses=L, a12=a12(d.ROCS.values, d[opp].values),
                         mean_improvement=imp.mean(), a12_ci_low=np.percentile(boot_a, 2.5),
                         a12_ci_high=np.percentile(boot_a, 97.5), imp_ci_low=np.percentile(boot_m, 2.5),
                         imp_ci_high=np.percentile(boot_m, 97.5), p_family_wilcoxon=p_fam))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "family_stats.csv", index=False)

    # Friedman and Nemenyi on family means (k = 11 methods, N = 9 families)
    fam = d.groupby("workflow")[METHODS].mean()
    ranks = fam.rank(axis=1).mean().sort_values()
    stat = friedmanchisquare(*[fam[m] for m in METHODS])
    k, n = len(METHODS), len(fam)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    run_ranks = d[METHODS].rank(axis=1).mean().sort_values()
    with open(OUT / "family_stats.md", "w") as f:
        f.write("# Family-level statistics\n\n")
        f.write("## ROCS against opponents, 2,430 runs, two-level bootstrap (families, then runs)\n\n")
        f.write("| Opponent | Wins | Ties | Losses | Paired A12 [95% CI] | Mean improvement % [95% CI] | p Wilcoxon on 9 family means |\n|---|---|---|---|---|---|---|\n")
        for _, r in table[table.family == "ALL"].iterrows():
            f.write(f"| {r.opponent} | {r.wins} | {r.ties} | {r.losses} | {r.a12:.3f} [{r.a12_ci_low:.3f}, {r.a12_ci_high:.3f}] | "
                    f"{r.mean_improvement:.2f} [{r.imp_ci_low:.2f}, {r.imp_ci_high:.2f}] | {r.p_family_wilcoxon:.3g} |\n")
        f.write("\n## Per family: ROCS against EZ+GA\n\n| Family | Wins | Ties | Losses | A12 | Improvement % |\n|---|---|---|---|---|---|\n")
        for _, r in table[(table.opponent == "EZ_GA") & (table.family != "ALL")].iterrows():
            f.write(f"| {r.family} | {r.wins} | {r.ties} | {r.losses} | {r.a12:.3f} | {r.mean_improvement:.2f} |\n")
        f.write(f"\n## Friedman test on family means (k = {k}, N = {n})\n\n")
        f.write(f"chi-square {stat.statistic:.2f}, p {stat.pvalue:.3g}, critical difference Nemenyi {cd:.2f}\n\n")
        f.write("| Method | Mean rank (families) | Mean rank (runs) |\n|---|---|---|\n")
        for m in ranks.index:
            f.write(f"| {m} | {ranks[m]:.2f} | {run_ranks[m]:.2f} |\n")
    print(open(OUT / "family_stats.md").read())


if __name__ == "__main__":
    main()
