> EXPLORATORY, post-registration learning curves per cluster (calib replay of B0's own inputs, each adapter at its cluster's call sites). Outside the preregistered plan; not a test-split result; never read by a verdict. Fixed epochs: rows and optimiser steps vary together. One seed per point.

## c0

| call site | point | calls (questions, distinct prompts; routed to the cluster) | J2 score | primary (SPEC 6.1) | parsed ok | latency p50 / p95 ms | tokens in / out per call |
|---|---|---|---|---|---|---|---|
| filter_column | 0% (base) | 13138 (200, 13138; 13128) | agreement 0.888 [0.877, 0.901] | recall kept 0.859 [0.836, 0.880]; agreement dropped 0.893 [0.878, 0.907]; kappa 0.594 [0.564, 0.623] | 1.000 [1.000, 1.000] | 7465 / 18325 | 4052 / 84 |
| filter_column | 25% | 13138 (200, 13138; 13128) | agreement 0.961 [0.954, 0.968] | recall kept 0.810 [0.782, 0.838]; agreement dropped 0.983 [0.978, 0.987]; kappa 0.816 [0.794, 0.836] | 1.000 [1.000, 1.000] | 18387 / 26023 | 4052 / 90 |
| filter_column | 50% | 13138 (200, 13138; 13128) | agreement 0.968 [0.962, 0.974] | recall kept 0.836 [0.809, 0.862]; agreement dropped 0.987 [0.983, 0.990]; kappa 0.850 [0.832, 0.869] | 1.000 [1.000, 1.000] | 17872 / 25835 | 4052 / 91 |
| filter_column | 100% | 13138 (200, 13138; 13128) | agreement 0.973 [0.968, 0.978] | recall kept 0.872 [0.850, 0.893]; agreement dropped 0.987 [0.984, 0.991]; kappa 0.874 [0.857, 0.890] | 1.000 [1.000, 1.000] | 18296 / 26169 | 4052 / 91 |
| filter_column | B4 (100%, cross-check) | 13138 (200, 13138; 13128) | agreement 0.972 [0.966, 0.977] | recall kept 0.869 [0.847, 0.891]; agreement dropped 0.987 [0.983, 0.990]; kappa 0.869 [0.853, 0.886] | 1.000 [1.000, 1.000] | 12186 / 24217 | 4052 / 91 |
| filter_column | teacher (self-replay, pilot) | 3311 (50, 3311; 3307) | agreement 0.991 [0.987, 0.994] | recall kept 0.972 [0.959, 0.985]; agreement dropped 0.994 [0.991, 0.997]; kappa 0.966 [0.956, 0.976] | 1.000 [1.000, 1.000] | 1957 / 3060 | 5891 / 139 |

## c1

| call site | point | calls (questions, distinct prompts; routed to the cluster) | J2 score | primary (SPEC 6.1) | parsed ok | latency p50 / p95 ms | tokens in / out per call |
|---|---|---|---|---|---|---|---|
| select_tables | 0% (base) | 200 (200, 200; 200) | agreement 0.820 [0.765, 0.870] | set recall 0.969 [0.954, 0.982] | 1.000 [1.000, 1.000] | 10115 / 19782 | 1241 / 154 |
| select_tables | 25% | 200 (200, 200; 200) | agreement 0.850 [0.800, 0.895] | set recall 0.965 [0.949, 0.979] | 1.000 [1.000, 1.000] | 4029 / 7783 | 1241 / 140 |
| select_tables | 50% | 200 (200, 200; 200) | agreement 0.865 [0.815, 0.910] | set recall 0.964 [0.948, 0.979] | 1.000 [1.000, 1.000] | 4042 / 7203 | 1241 / 139 |
| select_tables | 100% | 200 (200, 200; 200) | agreement 0.895 [0.850, 0.935] | set recall 0.972 [0.956, 0.985] | 1.000 [1.000, 1.000] | 3886 / 6864 | 1241 / 136 |
| select_tables | B4 (100%, cross-check) | 200 (200, 200; 200) | agreement 0.890 [0.840, 0.930] | set recall 0.973 [0.957, 0.985] | 1.000 [1.000, 1.000] | 12495 / 26964 | 1241 / 136 |
| select_tables | teacher (self-replay, pilot) | 50 (50, 50; 50) | agreement 0.980 [0.940, 1.000] | set recall 0.997 [0.990, 1.000] | 1.000 [1.000, 1.000] | 2510 / 4105 | 2010 / 239 |
| select_columns | 0% (base) | 200 (200, 200; 200) | agreement 0.395 [0.330, 0.460] | set recall 0.813 [0.787, 0.838] | 1.000 [1.000, 1.000] | 10387 / 21140 | 1125 / 165 |
| select_columns | 25% | 200 (200, 200; 200) | agreement 0.625 [0.560, 0.690] | set recall 0.912 [0.891, 0.932] | 1.000 [1.000, 1.000] | 4038 / 8547 | 1125 / 148 |
| select_columns | 50% | 200 (200, 200; 200) | agreement 0.675 [0.610, 0.740] | set recall 0.935 [0.916, 0.952] | 1.000 [1.000, 1.000] | 4042 / 8826 | 1125 / 149 |
| select_columns | 100% | 200 (200, 200; 200) | agreement 0.725 [0.660, 0.785] | set recall 0.951 [0.934, 0.967] | 1.000 [1.000, 1.000] | 4051 / 9462 | 1125 / 147 |
| select_columns | B4 (100%, cross-check) | 200 (200, 200; 200) | agreement 0.730 [0.670, 0.790] | set recall 0.946 [0.926, 0.962] | 1.000 [1.000, 1.000] | 12331 / 28648 | 1125 / 147 |
| select_columns | teacher (self-replay, pilot) | 50 (50, 50; 50) | agreement 0.860 [0.760, 0.940] | set recall 0.994 [0.986, 1.000] | 1.000 [1.000, 1.000] | 2221 / 3620 | 1797 / 247 |
| generate_candidate | 0% (base) | 199 (199, 199; 199) | EX 0.543 [0.472, 0.613] (teacher 0.688) | EX 0.543 [0.472, 0.613] (teacher 0.688) | 1.000 [1.000, 1.000] | 32295 / 94110 | 9929 / 685 |
| generate_candidate | 25% | 199 (199, 199; 199) | EX 0.568 [0.497, 0.638] (teacher 0.688) | EX 0.568 [0.497, 0.638] (teacher 0.688) | 1.000 [1.000, 1.000] | 11630 / 31029 | 9929 / 520 |
| generate_candidate | 50% | 199 (199, 199; 199) | EX 0.603 [0.533, 0.668] (teacher 0.688) | EX 0.603 [0.533, 0.668] (teacher 0.688) | 1.000 [1.000, 1.000] | 11413 / 26444 | 9929 / 504 |
| generate_candidate | 100% | 199 (199, 199; 199) | EX 0.613 [0.543, 0.683] (teacher 0.688) | EX 0.613 [0.543, 0.683] (teacher 0.688) | 1.000 [1.000, 1.000] | 11126 / 25313 | 9929 / 473 |
| generate_candidate | B4 (100%, cross-check) | 199 (199, 199; 199) | EX 0.618 [0.548, 0.683] (teacher 0.688) | EX 0.618 [0.548, 0.683] (teacher 0.688) | 1.000 [1.000, 1.000] | 34424 / 71127 | 9929 / 476 |
| revise | 0% (base) | 5 (5, 5; 5) | EX 0.000 [0.000, 0.000] (teacher 0.400) | EX 0.000 [0.000, 0.000] (teacher 0.400) | 1.000 [1.000, 1.000] | 58217 / 66739 | 3901 / 802 |
| revise | 25% | 5 (5, 5; 5) | EX 0.000 [0.000, 0.000] (teacher 0.400) | EX 0.000 [0.000, 0.000] (teacher 0.400) | 1.000 [1.000, 1.000] | 7834 / 32721 | 3901 / 424 |
| revise | 50% | 5 (5, 5; 5) | EX 0.000 [0.000, 0.000] (teacher 0.400) | EX 0.000 [0.000, 0.000] (teacher 0.400) | 1.000 [1.000, 1.000] | 9623 / 37771 | 3901 / 536 |
| revise | 100% | 5 (5, 5; 5) | EX 0.000 [0.000, 0.000] (teacher 0.400) | EX 0.000 [0.000, 0.000] (teacher 0.400) | 1.000 [1.000, 1.000] | 7625 / 20743 | 3901 / 320 |
| revise | B4 (100%, cross-check) | 5 (5, 5; 5) | EX 0.400 [0.000, 0.800] (teacher 0.400) | EX 0.400 [0.000, 0.800] (teacher 0.400) | 1.000 [1.000, 1.000] | 27174 / 37663 | 3901 / 374 |

## c2

| call site | point | calls (questions, distinct prompts; routed to the cluster) | J2 score | primary (SPEC 6.1) | parsed ok | latency p50 / p95 ms | tokens in / out per call |
|---|---|---|---|---|---|---|---|
| extract_keywords | 0% (base) | 200 (200, 200; 200) | agreement 0.000 [0.000, 0.000] | Jaccard 0.487 [0.463, 0.510] | 0.995 [0.985, 1.000] | 3494 / 11008 | 707 / 41 |
| extract_keywords | 25% | 200 (200, 200; 200) | agreement 0.025 [0.005, 0.050] | Jaccard 0.636 [0.616, 0.657] | 0.995 [0.985, 1.000] | 962 / 2338 | 707 / 80 |
| extract_keywords | 50% | 200 (200, 200; 200) | agreement 0.065 [0.030, 0.100] | Jaccard 0.686 [0.663, 0.709] | 1.000 [1.000, 1.000] | 993 / 2226 | 707 / 61 |
| extract_keywords | 100% | 200 (200, 200; 200) | agreement 0.110 [0.065, 0.155] | Jaccard 0.729 [0.705, 0.752] | 1.000 [1.000, 1.000] | 983 / 2289 | 707 / 61 |
| extract_keywords | B4 (100%, cross-check) | 200 (200, 200; 200) | agreement 0.125 [0.080, 0.170] | Jaccard 0.731 [0.708, 0.753] | 1.000 [1.000, 1.000] | 5259 / 15432 | 707 / 61 |
| extract_keywords | teacher (self-replay, pilot) | 50 (50, 50; 50) | agreement 0.440 [0.300, 0.580] | Jaccard 0.899 [0.862, 0.932] | 1.000 [1.000, 1.000] | 1741 / 5114 | 1082 / 98 |

## c3

| call site | point | calls (questions, distinct prompts; routed to the cluster) | J2 score | primary (SPEC 6.1) | parsed ok | latency p50 / p95 ms | tokens in / out per call |
|---|---|---|---|---|---|---|---|
| agent_ir | 0% (base) | 800 (200, 4; 800) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1276 / 2435 | 246 / 5 |
| agent_ir | B4 (100%, cross-check) | 800 (200, 4; 800) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 935 / 2099 | 246 / 4 |
| agent_ir | teacher (self-replay, pilot) | 200 (50, 4; 200) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1245 / 1549 | 377 / 17 |
| agent_ss | 0% (base) | 800 (200, 4; 800) | agreement 0.750 [0.750, 0.750] | agreement per distinct prompt 0.750 [0.750, 0.750] | 1.000 [1.000, 1.000] | 1095 / 2633 | 232 / 5 |
| agent_ss | B4 (100%, cross-check) | 800 (200, 4; 800) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1261 / 2492 | 232 / 5 |
| agent_ss | teacher (self-replay, pilot) | 200 (50, 4; 200) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1246 / 2076 | 345 / 47 |
| agent_cg | 0% (base) | 598 (200, 3; 598) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1093 / 7812 | 230 / 23 |
| agent_cg | B4 (100%, cross-check) | 598 (200, 3; 598) | agreement 1.000 [1.000, 1.000] | agreement per distinct prompt 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1187 / 2625 | 230 / 5 |
| agent_cg | teacher (self-replay, pilot) | 150 (50, 3; 150) | agreement 0.993 [0.980, 1.000] | agreement per distinct prompt 0.993 [0.980, 1.000] | 0.993 [0.980, 1.000] | 1343 / 2428 | 337 / 67 |

