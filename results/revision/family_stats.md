# Family-level statistics

## ROCS against opponents, 2,430 runs, two-level bootstrap (families, then runs)

| Opponent | Wins | Ties | Losses | Paired A12 [95% CI] | Mean improvement % [95% CI] | p Wilcoxon on 9 family means |
|---|---|---|---|---|---|---|
| EZ | 1219 | 1211 | 0 | 0.751 [0.667, 0.835] | 0.70 [0.10, 1.64] | 0.00391 |
| EZ_GA | 671 | 1301 | 458 | 0.544 [0.464, 0.615] | -0.29 [-0.89, 0.03] | 0.496 |
| EZ_PCSA | 866 | 1279 | 285 | 0.620 [0.573, 0.673] | -0.07 [-0.40, 0.18] | 0.82 |
| GA | 2249 | 76 | 105 | 0.941 [0.897, 0.975] | 4.93 [0.31, 11.52] | 0.00391 |
| PEFT | 2428 | 2 | 0 | 1.000 [0.999, 1.000] | 14.66 [8.92, 21.34] | 0.00391 |

## Per family: ROCS against EZ+GA

| Family | Wins | Ties | Losses | A12 | Improvement % |
|---|---|---|---|---|---|
| Blast | 28 | 214 | 28 | 0.500 | -0.23 |
| Cycles | 78 | 123 | 69 | 0.517 | -0.06 |
| Bwa | 38 | 224 | 8 | 0.556 | 0.00 |
| Epigenomics | 70 | 21 | 179 | 0.298 | -2.02 |
| Genome | 111 | 140 | 19 | 0.670 | 0.15 |
| Seismology | 46 | 214 | 10 | 0.567 | 0.02 |
| Montage | 101 | 61 | 108 | 0.487 | -0.54 |
| Srasearch | 60 | 188 | 22 | 0.570 | 0.02 |
| Soykb | 139 | 116 | 15 | 0.730 | 0.03 |

## Friedman test on family means (k = 11, N = 9)

chi-square 84.79, p 5.75e-14, critical difference Nemenyi 5.03

| Method | Mean rank (families) | Mean rank (runs) |
|---|---|---|
| EZ_GA | 1.56 | 2.42 |
| ROCS | 1.78 | 2.26 |
| EZ_PCSA | 2.67 | 2.76 |
| EZ | 4.44 | 3.46 |
| GA | 5.22 | 6.14 |
| PCSA | 5.67 | 6.52 |
| CSA | 7.67 | 7.86 |
| CSA_OBL | 7.89 | 7.84 |
| PSO | 8.11 | 7.78 |
| HEFT | 10.00 | 8.15 |
| PEFT | 11.00 | 10.80 |
