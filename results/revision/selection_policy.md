# Selection policy evaluation

| policy                                  |   refine_share |   balanced_accuracy |   mean_improvement |   retained |   refinement_seconds |   time_saved |
|:----------------------------------------|---------------:|--------------------:|-------------------:|-----------:|---------------------:|-------------:|
| KEEP                                    |          0     |               0.5   |              0     |      0     |                0     |      100     |
| ALWAYS                                  |        100     |               0.5   |              0.69  |    100     |             2241.69  |        0     |
| TREE (leave-one-family-out)             |         41.481 |               0.607 |              0.558 |     80.865 |              816.627 |       63.571 |
| TREE (E6 GroupKFold, as in the article) |         42.387 |               0.65  |              0.572 |     82.89  |              795.563 |       64.511 |
| GAP threshold (leave-one-family-out)    |         52.222 |               0.688 |              0.607 |     87.957 |             1008.47  |       55.013 |
| GAP fixed C_EZ/LB > 1.01                |         70.123 |               0.669 |              0.688 |     99.787 |             1362.77  |       39.208 |
| GAP fixed C_EZ/LB > 1.05                |         53.292 |               0.732 |              0.668 |     96.898 |             1018.63  |       54.56  |

Gap threshold chosen for each held-out family: {'Blast': 1.049, 'Bwa': 1.049, 'Cycles': 1.049, 'Epigenomics': 1.049, 'Genome': 1.196, 'Montage': 1.05, 'Seismology': 1.049, 'Soykb': 1.049, 'Srasearch': 1.049}

## Per held-out family

| family      | policy   |   refine_share |   error_rate |   mean_improvement |   retained |   time_saved |
|:------------|:---------|---------------:|-------------:|-------------------:|-----------:|-------------:|
| Blast       | TREE     |           5.19 |        10.37 |               0    |       0.22 |        86.96 |
| Blast       | GAP      |          37.04 |        31.85 |               0.06 |      98.42 |        78.8  |
| Blast       | ALWAYS   |         100    |        94.81 |               0.07 |     100    |         0    |
| Bwa         | TREE     |         100    |       100    |               0    |     100    |         0    |
| Bwa         | GAP      |         100    |       100    |               0    |     100    |         0    |
| Bwa         | ALWAYS   |         100    |       100    |               0    |     100    |         0    |
| Cycles      | TREE     |           3.7  |        16.67 |               0    |       1.14 |        99.08 |
| Cycles      | GAP      |          25.56 |        20.74 |               0.13 |      61.07 |        96.51 |
| Cycles      | ALWAYS   |         100    |        87.04 |               0.21 |     100    |         0    |
| Epigenomics | TREE     |         100    |        55.93 |               3.9  |     100    |         0    |
| Epigenomics | GAP      |         100    |        55.93 |               3.9  |     100    |         0    |
| Epigenomics | ALWAYS   |         100    |        55.93 |               3.9  |     100    |         0    |
| Genome      | TREE     |          33.33 |        15.93 |               1.02 |      99.38 |        66.56 |
| Genome      | GAP      |          11.48 |        12.59 |               0.46 |      45.13 |        99.09 |
| Genome      | ALWAYS   |         100    |        82.59 |               1.02 |     100    |         0    |
| Montage     | TREE     |           0    |        22.59 |               0    |       0    |       100    |
| Montage     | GAP      |          76.3  |        55.19 |               0.81 |      94.2  |        34.47 |
| Montage     | ALWAYS   |         100    |        77.41 |               0.86 |     100    |         0    |
| Seismology  | TREE     |          24.07 |        24.07 |               0.02 |      51.78 |        94.16 |
| Seismology  | GAP      |          11.11 |        11.11 |               0.01 |      46.22 |        99.2  |
| Seismology  | ALWAYS   |         100    |       100    |               0.03 |     100    |         0    |
| Soykb       | TREE     |          74.44 |        74.44 |               0.03 |      68.42 |        61.95 |
| Soykb       | GAP      |         100    |       100    |               0.04 |     100    |         0    |
| Soykb       | ALWAYS   |         100    |       100    |               0.04 |     100    |         0    |
| Srasearch   | TREE     |          32.59 |        31.48 |               0.05 |      73.12 |        68.55 |
| Srasearch   | GAP      |           8.52 |         8.15 |               0.03 |      48.56 |        99.42 |
| Srasearch   | ALWAYS   |         100    |        97.41 |               0.07 |     100    |         0    |

## Leaf support of the full-data tree (2,430 instances)

|--- ez_over_lb <= 1.05
|   |--- ez_over_lb <= 1.01
|   |   |--- parallelism <= 1.46
|   |   |   |--- class: 0
|   |   |--- parallelism >  1.46
|   |   |   |--- class: 0
|   |--- ez_over_lb >  1.01
|   |   |--- relative_width <= 0.49
|   |   |   |--- class: 1
|   |   |--- relative_width >  0.49
|   |   |   |--- class: 0
|--- ez_over_lb >  1.05
|   |--- edges_per_task <= 2.96
|   |   |--- top10_transfer_share <= 0.18
|   |   |   |--- class: 0
|   |   |--- top10_transfer_share >  0.18
|   |   |   |--- class: 1
|   |--- edges_per_task >  2.96
|   |   |--- ez_over_lb <= 3.30
|   |   |   |--- class: 0
|   |   |--- ez_over_lb >  3.30
|   |   |   |--- class: 1

|   leaf |   n |   positives |   predicted |
|-------:|----:|------------:|------------:|
|      3 |   2 |           0 |           0 |
|      4 | 849 |           0 |           0 |
|      6 |  83 |          14 |           1 |
|      7 | 198 |           0 |           0 |
|     10 | 234 |           0 |           0 |
|     11 | 677 |         268 |           1 |
|     13 | 386 |           0 |           0 |
|     14 |   1 |           1 |           1 |

## Guaranteed condition C_EZ/LB <= 1.01

726 instances, positive labels 0, largest gain 0.611%. Grouped balanced accuracy of the tree outside this subset: 0.576

## Trees of the GroupKFold folds

Fold 0, tested on ['Epigenomics', 'Srasearch']

```
|--- ccr <= 3.00
|   |--- depth <= 6.00
|   |   |--- chain_task_share <= 0.01
|   |   |   |--- class: 0
|   |   |--- chain_task_share >  0.01
|   |   |   |--- class: 0
|   |--- depth >  6.00
|   |   |--- top10_transfer_share <= 0.23
|   |   |   |--- class: 0
|   |   |--- top10_transfer_share >  0.23
|   |   |   |--- class: 1
|--- ccr >  3.00
|   |--- relative_width <= 0.99
|   |   |--- top10_transfer_share <= 0.18
|   |   |   |--- class: 0
|   |   |--- top10_transfer_share >  0.18
|   |   |   |--- class: 1
|   |--- relative_width >  0.99
|   |   |--- edges_per_task <= 1.00
|   |   |   |--- class: 0
|   |   |--- edges_per_task >  1.00
|   |   |   |--- class: 0
```

Fold 1, tested on ['Cycles', 'Soykb']

```
|--- ez_over_lb <= 1.05
|   |--- parallelism <= 26.97
|   |   |--- parallelism <= 26.12
|   |   |   |--- class: 0
|   |   |--- parallelism >  26.12
|   |   |   |--- class: 1
|   |--- parallelism >  26.97
|   |   |--- class: 0
|--- ez_over_lb >  1.05
|   |--- edges_per_task <= 2.97
|   |   |--- ez_over_lb <= 1.43
|   |   |   |--- class: 1
|   |   |--- ez_over_lb >  1.43
|   |   |   |--- class: 1
|   |--- edges_per_task >  2.97
|   |   |--- ez_over_lb <= 3.30
|   |   |   |--- class: 0
|   |   |--- ez_over_lb >  3.30
|   |   |   |--- class: 1
```

Fold 2, tested on ['Bwa', 'Seismology']

```
|--- ez_over_lb <= 1.05
|   |--- max_out_degree <= 0.00
|   |   |--- ez_over_lb <= 1.02
|   |   |   |--- class: 0
|   |   |--- ez_over_lb >  1.02
|   |   |   |--- class: 1
|   |--- max_out_degree >  0.00
|   |   |--- ez_over_lb <= 1.03
|   |   |   |--- class: 0
|   |   |--- ez_over_lb >  1.03
|   |   |   |--- class: 0
|--- ez_over_lb >  1.05
|   |--- top10_transfer_share <= 0.18
|   |   |--- class: 0
|   |--- top10_transfer_share >  0.18
|   |   |--- ez_over_lb <= 1.43
|   |   |   |--- class: 1
|   |   |--- ez_over_lb >  1.43
|   |   |   |--- class: 1
```

Fold 3, tested on ['Blast', 'Montage']

```
|--- relative_width <= 0.61
|   |--- ez_over_lb <= 1.01
|   |   |--- class: 0
|   |--- ez_over_lb >  1.01
|   |   |--- edges_per_task <= 1.56
|   |   |   |--- class: 1
|   |   |--- edges_per_task >  1.56
|   |   |   |--- class: 0
|--- relative_width >  0.61
|   |--- top10_transfer_share <= 0.60
|   |   |--- class: 0
|   |--- top10_transfer_share >  0.60
|   |   |--- ez_over_lb <= 1.05
|   |   |   |--- class: 0
|   |   |--- ez_over_lb >  1.05
|   |   |   |--- class: 1
```

Fold 4, tested on ['Genome']

```
|--- relative_width <= 0.47
|   |--- ez_over_lb <= 1.39
|   |   |--- ez_over_lb <= 1.08
|   |   |   |--- class: 0
|   |   |--- ez_over_lb >  1.08
|   |   |   |--- class: 1
|   |--- ez_over_lb >  1.39
|   |   |--- top10_transfer_share <= 0.34
|   |   |   |--- class: 0
|   |   |--- top10_transfer_share >  0.34
|   |   |   |--- class: 1
|--- relative_width >  0.47
|   |--- measured_ccr <= 3.00
|   |   |--- depth <= 6.00
|   |   |   |--- class: 0
|   |   |--- depth >  6.00
|   |   |   |--- class: 0
|   |--- measured_ccr >  3.00
|   |   |--- relative_width <= 0.99
|   |   |   |--- class: 1
|   |   |--- relative_width >  0.99
|   |   |   |--- class: 0
```

## Full-data tree applied to the E3 fleets

| policy         |   refine_share |   balanced_accuracy |   mean_improvement |   retained |   refinement_seconds |   time_saved | fleet                 |
|:---------------|---------------:|--------------------:|-------------------:|-----------:|---------------------:|-------------:|:----------------------|
| TREE full-data |             34 |               0.801 |              0.958 |     91.009 |               21.848 |       64.653 | e3_8vm                |
| ALWAYS         |            100 |               0.5   |              1.053 |    100     |               61.809 |        0     | e3_8vm                |
| TREE full-data |             67 |               0.701 |              1.324 |     99.293 |               38.826 |       31.453 | e3_32vm               |
| ALWAYS         |            100 |               0.5   |              1.334 |    100     |               56.642 |        0     | e3_32vm               |
| TREE full-data |             57 |               0.791 |              0.737 |     98.286 |               31.119 |       39.427 | e3_low_heterogeneity  |
| ALWAYS         |            100 |               0.5   |              0.75  |    100     |               51.375 |        0     | e3_low_heterogeneity  |
| TREE full-data |             52 |               0.727 |              0.993 |     90.188 |               27.88  |       44.82  | e3_high_heterogeneity |
| ALWAYS         |            100 |               0.5   |              1.101 |    100     |               50.525 |        0     | e3_high_heterogeneity |
