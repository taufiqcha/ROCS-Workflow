# Values reported in Section 5

## 5.1 Independent tasks (E5)

Median makespan relative to the lower bound (Table 3 and Fig. 2)

| Dataset | Tasks | MIN_MIN | LPT | CSA | CSA_OBL | CSA_ROJ | GA | ROCS_LPT |
|---|---|---|---|---|---|---|---|---|
| rnaseq | 1000 | 1.278 | 1.015 | 1.142 | 1.130 | 1.053 | 1.015 | 1.015 |
| rnaseq | 3000 | 1.105 | 1.007 | 1.071 | 1.055 | 1.013 | 1.007 | 1.007 |
| rnaseq | 5000 | 1.047 | 1.004 | 1.040 | 1.042 | 1.009 | 1.004 | 1.004 |
| rnaseq | 7000 | 1.053 | 1.005 | 1.035 | 1.031 | 1.008 | 1.005 | 1.005 |
| rnaseq | 10000 | 1.022 | 1.005 | 1.031 | 1.041 | 1.009 | 1.005 | 1.005 |
| simple | 1000 | 1.041 | 1.010 | 1.071 | 1.065 | 1.016 | 1.010 | 1.010 |
| simple | 3000 | 1.014 | 1.003 | 1.032 | 1.027 | 1.004 | 1.003 | 1.003 |
| simple | 5000 | 1.008 | 1.002 | 1.022 | 1.022 | 1.003 | 1.002 | 1.002 |
| simple | 7000 | 1.005 | 1.001 | 1.021 | 1.020 | 1.002 | 1.001 | 1.001 |
| simple | 10000 | 1.006 | 1.002 | 1.017 | 1.017 | 1.001 | 1.002 | 1.002 |
| stratified | 1000 | 2.365 | 1.614 | 1.913 | 1.834 | 1.815 | 1.614 | 1.614 |
| stratified | 3000 | 1.370 | 1.000 | 1.206 | 1.224 | 1.095 | 1.000 | 1.000 |
| stratified | 5000 | 1.228 | 1.000 | 1.131 | 1.129 | 1.035 | 1.000 | 1.000 |
| stratified | 7000 | 1.147 | 1.000 | 1.100 | 1.089 | 1.013 | 1.000 | 1.000 |
| stratified | 10000 | 1.099 | 1.000 | 1.070 | 1.070 | 1.002 | 1.000 | 1.000 |

- LPT within 1.5% of the bound: 14 of 15 configurations
- ROCS from LPT shorter than LPT: 0 of 450 runs
- GA from LPT tied with LPT: 441 of 450 runs
- CSA-OBL against CSA, p per dataset: 0.11 to 0.28
- CSA-ROJ against CSA, largest p per dataset: 2.6e-10
- CSA-ROJ shorter than CSA: 412 of 450 runs
- Median gap to the bound from 3,000 tasks: CSA-ROJ 0.1% to 9.5%, CSA 1.7% to 20.6%

## 5.2 Workflow benchmark (E1)

Friedman chi-square 20789, p 0, Nemenyi critical difference 0.306 (Fig. 3)

| Method | Average rank |
|---|---|
| ROCS | 2.26 |
| EZ_GA | 2.42 |
| EZ_PCSA | 2.76 |
| EZ | 3.46 |
| GA | 6.14 |
| PCSA | 6.52 |
| PSO | 7.78 |
| CSA_OBL | 7.84 |
| CSA | 7.86 |
| HEFT | 8.15 |
| PEFT | 10.80 |

- EZ over HEFT and PEFT on Montage and Epigenomics (Fig. 4a): CCR 1 19.5% to 25.7%, CCR 5 48.3% to 57.9%
- Mean ROCS improvement (Fig. 4b): CCR 0.1 0.14%, CCR 5 1.47%, largest workflow mean 3.90% (Epigenomics)
- ROCS more than 1% better than the best heuristic at CCR 5: 14.0% of runs

Pairwise comparison of ROCS (Table 4)

| Opponent | Wins | Ties | Losses | p Holm | Paired A12 |
|---|---|---|---|---|---|
| EZ | 1219 | 1211 | 0 | 8.4e-233 | 0.751 |
| EZ_GA | 671 | 1301 | 458 | 6.6e-06 | 0.544 |
| EZ_PCSA | 866 | 1279 | 285 | 4.2e-49 | 0.620 |

- Mean run time, 200 tasks: ROCS 0.06 s, EZ+GA 0.22 s
- Mean run time, 500 tasks: ROCS 0.40 s, EZ+GA 0.98 s
- Mean run time, 1000 tasks: ROCS 2.30 s, EZ+GA 4.28 s
- Mean makespan reduction of ROCS on Montage and Epigenomics at CCR 1 and 5: PEFT 42.5%, HEFT 41.7%, CSA_OBL 39.5%, GA 30.7%

## 5.3 Sensitivity (E2 and E3)

Mean improvement over the best heuristic by budget (Fig. 5)

| Budget | ROCS | EZ_GA | EZ_PCSA | GA | CSA_OBL |
|---|---|---|---|---|---|
| 20n | 2.3 | 2.3 | 1.6 | -42.4 | -50.9 |
| 100n | 3.5 | 4.0 | 3.4 | -21.7 | -50.6 |
| 500n | 4.3 | 4.4 | 4.1 | -13.3 | -50.2 |

| Fleet | EZ over HEFT and PEFT | ROCS over best heuristic |
|---|---|---|
| 16 VMs, 400/500/600 | 16.3 | 1.9 |
| 8 VMs | 13.6 | 1.1 |
| 32 VMs | 18.1 | 1.3 |
| 16 VMs, 450/500/550 | 15.3 | 0.8 |
| 16 VMs, 250/500/1000 | 20.3 | 1.1 |

## 5.4 Optimality gap (E4) and dual encoding (E7)

- CP-SAT proved optimality on 9 of 36 instances and returned the best known solution on 32
- Median gap to the proven optimum: HEFT 5.9%, EZ 5.9%, ROCS 5.1%
- Largest gap to the proven optimum: EZ 17.1%, GA 17.1%, CSA_OBL 17.1%, EZ_GA 17.1%, EZ_PCSA 17.1%, ROCS 17.1%
- Median gap of DCSA to the best known solution (Fig. 6): 20 3.1%, 100 2.7%, 500 1.8%, 2000 1.4%
- DCSA at 2000n shorter than EZ+PCSA on 30 instances and shorter than CP-SAT on 4

Dual encoding at 100n, improvement over EZ (Table 5)

| Method | Mean (%) | Median (%) | DCSA wins | DCSA losses | p |
|---|---|---|---|---|---|
| GA_DUAL | 3.81 | 0.87 | 40 | 88 | 4.39e-06 |
| EZ_GA | 3.54 | 0.69 | 59 | 66 | 0.459 |
| DCSA | 3.03 | 0.76 | | | |
| DCSA_NO_KEYS | 3.07 | 0.51 | 81 | 38 | 0.000287 |
| DCSA_NO_OBL | 2.84 | 0.77 | 49 | 59 | 0.492 |
| ROCS | 2.98 | 0.46 | 92 | 40 | 5.81e-07 |
| EZ_PCSA | 2.74 | 0.51 | 81 | 41 | 0.000148 |

## 5.5 Selection rule (E6)

- Balanced accuracy with grouped cross-validation: 0.65
- ROCS invoked on 42.4% of the instances, refinement time saved 64.5%
- Mean improvement over the best heuristic: framework 0.57%, ROCS on every instance 0.69%, retained 83%
- Runs with more than 5% improvement selected: 54 of 70
