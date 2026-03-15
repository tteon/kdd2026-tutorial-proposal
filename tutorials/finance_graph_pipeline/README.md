# Finance Graph Pipeline Tutorial

This tutorial package demonstrates the minimal graph pipeline needed for the
core tutorial claim:

1. Profile selection
2. Ontology-constrained extraction
3. Entity linking
4. Graph construction
5. Graph quality analysis
6. Quality-aware evidence selection
7. Answer generation

It is intentionally lightweight so you can inspect the flow end to end. The
pipeline supports both heuristic mode and OpenAI Agents SDK mode for profile
selection, ontology-constrained extraction, and answer generation.

## Minimal Problem-Solution-Evidence Frame

### Problem

Question-only or reference-only retrieval is often insufficient for domain-specific, relation-heavy questions.
However, graph-grounded generation only helps if the graph is constructed with the right
ontology profile, extraction quality, and graph quality.

### Solution

We keep only the minimum end-to-end pipeline needed to test this claim:

- FIBO-like profile selection
- Profile-constrained extraction
- Entity linking
- Graph construction
- Graph quality analysis
- Quality-aware evidence filtering
- Answer generation

### What Should Be Measured

- `profile_selection_accuracy`
- `ontology_constrained_extraction_f1`
- `query_support_path_coverage`
- `answer_quality_delta`

These four metrics are enough to evaluate whether improvements in upstream graph
construction transfer to downstream answers.

Current measurement details:

- `query_support_path_coverage` is question-intent-specific, not just profile-level relation presence
- the main comparison uses one shared answer agent with a fixed prompt, fixed model, and fixed context budget
- `graph` answers are generated from a compact evidence bundle built from selected triples and serialized as structured text
- extraction can be evaluated against the small manual gold subset before broader normalization work

The evaluation output separates:

- auxiliary non-graph baselines: `question_only_baseline`, `reference_only_baseline`
- primary ontology-guided baselines: graph variants in the main `comparison_table`

## Contents

- `data/sample_finance_dataset.json`: Small finance dataset slice
- `data/finder_manual_gold_subset.json`: 30-example manual gold subset for FinDER top-3 categories
- `framework/`: Reusable Python modules
- `notebooks/finance_graph_pipeline_tutorial.ipynb`: Notebook-style walkthrough

## Run The Framework Demo

```bash
python3 tutorials/finance_graph_pipeline/framework/demo.py
```

Run the baseline evaluation harness:

```bash
python3 tutorials/finance_graph_pipeline/framework/evaluate.py
```

Run the minimal FinDER experiment runner:

```bash
python3 scripts/run_finder_experiment.py --per-category-limit 10
```

Use the legacy code-based answer synthesis only as a supplemental baseline:

```bash
python3 scripts/run_finder_experiment.py --per-category-limit 10 --answer-mode heuristic_synthesis
```

Run it against the manual gold subset only:

```bash
python3 scripts/run_finder_experiment.py \
  --manual-gold-path tutorials/finance_graph_pipeline/data/finder_manual_gold_subset.json \
  --manual-gold-only \
  --sample-size 30
```

Evaluate an existing run against the manual gold subset:

```bash
python3 scripts/evaluate_manual_gold_subset.py \
  --run-id <run_id> \
  --manual-gold-path tutorials/finance_graph_pipeline/data/finder_manual_gold_subset.json
```

Build a proposal-oriented report with mean, 95% bootstrap CI, and category breakdown:

```bash
python3 scripts/build_proposal_metrics_report.py \
  --run-id <primary_run_id> \
  --manual-eval-run-id <manual_gold_run_id>
```

The run summary reports `comparison_table` for ontology-guided baselines and
`auxiliary_comparison_table` for the two non-graph sanity checks.

## Design Notes

- `FiboProfileAgent`: selects a constrained FIBO-like profile for each document
- `ExtractionAgent`: performs profile-conditioned extraction
- `AnswerAgent`: uses the same answer prompt across baselines while changing only the context bundle
- `EntityLinker`: normalizes aliases into canonical graph nodes
- `GraphQualityAnalyzer`: computes global graph quality and question-intent-specific query-support path coverage
- `EvidenceSelector`: keeps only profile-matched, query-relevant evidence

This scaffold is designed so you can swap between heuristic and agent-backed
components without changing the minimal evaluation frame.
