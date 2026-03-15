# Table 1. Graph-Centered Shared Answer-Agent Results

| Baseline | Answer Quality | Delta vs Graph | Profile Acc | Extraction F1 | Coverage | Schema Conformance | Fallback Error Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Graph | 0.175 [0.165, 0.184] | 0.000 | - | 0.126 [0.103, 0.151] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Graph+Profile | 0.175 [0.166, 0.184] | 0.000 | 0.900 [0.847, 0.940] | 0.104 [0.082, 0.130] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Graph+Constraint | 0.318 [0.297, 0.338] | 0.143 | 0.900 [0.847, 0.940] | 0.083 [0.065, 0.104] | 0.589 [0.537, 0.638] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| Full | 0.321 [0.300, 0.341] | 0.146 | 0.900 [0.847, 0.940] | 0.083 [0.065, 0.104] | 0.589 [0.537, 0.638] | 1.000 [1.000, 1.000] | 0.007 [0.000, 0.020] |

Notes:

- This main table is restricted to graph baselines to keep the narrative centered on ontology-guided ablations.
- Values are overall means on the `50/category` shared answer-agent run.
- Brackets denote 95% bootstrap CI when available.
- `Delta vs Graph` uses `graph_without_ontology_constraints` as the anchor baseline.