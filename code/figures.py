"""Figures 2 to 6 of the article from the archived results.

Figures 1 and 7 are diagrams. Two main colors are used, and every series also differs by marker
or line style so that the figures read in grayscale.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import studentized_range

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = "#1F4E79", "#C55A11", "#8C8C8C"
ONE_COLUMN, TWO_COLUMNS = 3.5, 7.2
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
LABEL = {"EZ_GA": "EZ+GA", "EZ_PCSA": "EZ+PCSA", "CSA_OBL": "CSA-OBL", "MIN_MIN": "Min-Min", "CSA_ROJ": "CSA-ROJ"}
PROPOSED = {"ROCS", "DCSA_20", "DCSA_100", "DCSA_500", "DCSA_2000"}
HEURISTICS = ["HEFT", "PEFT", "EZ"]


def label(m):
    if m.startswith("DCSA_"):
        return f"DCSA {m[5:]}N"
    return LABEL.get(m, m)


def independent():
    d = pd.read_csv(R / "e5.csv")
    style = {"MIN_MIN": (GREY, "s", "--"), "LPT": (BLUE, "o", "-"), "CSA": (ORANGE, "^", ":"),
             "CSA_OBL": (ORANGE, "v", "--"), "CSA_ROJ": (ORANGE, "D", "-")}
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COLUMNS, 2.3))
    for ax, dataset, title in zip(axes, ["simple", "stratified", "rnaseq"], ["(a) Simple", "(b) Stratified", "(c) Rnaseq"]):
        g = d[d.dataset == dataset]
        for m, (color, marker, line) in style.items():
            v = (g[m] / g.lower_bound).groupby(g["size"]).median()
            ax.plot(v.index / 1000, v.values, marker=marker, ls=line, ms=3.5, lw=1.1, color=color, label=label(m))
        ax.set_title(title)
        ax.set_xlabel("Tasks (thousands)")
    axes[0].set_ylabel("Makespan / lower bound")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(F / "Figure_2.pdf")
    plt.close(fig)


def friedman(e1):
    methods = HEURISTICS + ["PSO", "CSA", "CSA_OBL", "GA", "PCSA", "ROCS", "EZ_GA", "EZ_PCSA"]
    ranks = e1[methods].rank(axis=1).mean().sort_values()
    k, n = len(methods), len(e1)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    fig, ax = plt.subplots(figsize=(ONE_COLUMN, 2.6))
    y = np.arange(len(ranks))
    colors = [ORANGE if m in PROPOSED else BLUE if m in HEURISTICS else GREY for m in ranks.index]
    ax.barh(y, ranks.values, color=colors, height=0.65)
    ax.set_yticks(y, [label(m) for m in ranks.index])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank")
    ax.errorbar(ranks.values[0] + cd / 2, len(ranks) - 0.2, xerr=cd / 2, color="k", capsize=2.5, lw=0.9)
    ax.text(ranks.values[0] + cd + 0.15, len(ranks) - 0.2, "CD", va="center")
    fig.savefig(F / "Figure_3.pdf")
    plt.close(fig)


def heatmap(e1):
    e1 = e1.copy()
    e1["ez"] = (e1[["HEFT", "PEFT"]].min(1) - e1.EZ) / e1[["HEFT", "PEFT"]].min(1) * 100
    e1["rocs"] = (e1[HEURISTICS].min(1) - e1.ROCS) / e1[HEURISTICS].min(1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMNS, 2.9))
    for ax, column, title, color in [(axes[0], "ez", "(a) EZ over HEFT and PEFT", BLUE),
                                     (axes[1], "rocs", "(b) ROCS over best heuristic", ORANGE)]:
        T = e1.pivot_table(index="workflow", columns="ccr", values=column, aggfunc="mean")
        image = ax.imshow(T.values, cmap=LinearSegmentedColormap.from_list("c", ["#FFFFFF", color]), aspect="auto")
        ax.set_xticks(range(T.shape[1]), [f"CCR {c:g}" for c in T.columns])
        ax.set_yticks(range(T.shape[0]), T.index)
        dark = np.nanmax(T.values) * 0.6
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                v = T.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > dark else "black")
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=ax, shrink=0.85, label="Mean improvement (%)")
    fig.tight_layout()
    fig.savefig(F / "Figure_4.pdf")
    plt.close(fig)


def budget(e1):
    subset = e1[(e1["size"] == 500) & (e1.seed < 10) & e1.workflow.isin(["Epigenomics", "Montage", "Genome", "Cycles"])
                & e1.ccr.isin([1.0, 5.0])]
    data = {20: subset, 100: pd.read_csv(R / "e2_100n.csv"), 500: pd.read_csv(R / "e2_500n.csv")}
    style = {"ROCS": (ORANGE, "o", "-"), "EZ_GA": (BLUE, "s", "-"), "EZ_PCSA": (BLUE, "^", "--"),
             "GA": (GREY, "s", ":"), "CSA_OBL": (GREY, "v", "--")}
    fig, ax = plt.subplots(figsize=(ONE_COLUMN, 2.4))
    for m, (color, marker, line) in style.items():
        values = [((d[HEURISTICS].min(1) - d[m]) / d[HEURISTICS].min(1) * 100).mean() for d in data.values()]
        ax.plot(list(data), values, marker=marker, ls=line, ms=3.5, lw=1.1, color=color, label=label(m))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks(list(data), [f"{b}N" for b in data])
    ax.minorticks_off()
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Mean improvement (%)")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(F / "Figure_5.pdf")
    plt.close(fig)


def optimality_gap():
    d = pd.read_csv(R / "e4.csv")
    methods = ["HEFT", "EZ", "GA", "CSA_OBL", "EZ_GA", "EZ_PCSA", "ROCS", "DCSA_20", "DCSA_100", "DCSA_500", "DCSA_2000"]
    best = d[["cpsat_best"] + methods].min(1)
    fig, ax = plt.subplots(figsize=(TWO_COLUMNS, 2.4))
    gaps = [((d[m] - best) / best * 100).clip(upper=40).values for m in methods]
    boxes = ax.boxplot(gaps, patch_artist=True, widths=0.55, flierprops={"markersize": 2, "markeredgecolor": GREY},
                       medianprops={"color": "k", "lw": 1})
    for box, m in zip(boxes["boxes"], methods):
        box.set_facecolor(ORANGE if m in PROPOSED else BLUE if m in HEURISTICS else "#D9D9D9")
        box.set_edgecolor("k")
        box.set_linewidth(0.6)
    ax.set_xticks(range(1, len(methods) + 1), [label(m).replace("DCSA ", "DCSA\n") for m in methods])
    ax.set_ylabel("Gap to best known (%)")
    fig.savefig(F / "Figure_6.pdf")
    plt.close(fig)


if __name__ == "__main__":
    e1 = pd.read_csv(R / "e1.csv")
    independent()
    friedman(e1)
    heatmap(e1)
    budget(e1)
    optimality_gap()
