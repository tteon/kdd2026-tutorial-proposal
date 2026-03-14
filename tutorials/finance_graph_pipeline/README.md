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

It is intentionally lightweight and heuristic-driven so you can inspect the flow
without depending on external LLM or graph infrastructure.

## Minimal Problem-Solution-Evidence Frame

### Problem

Text-only retrieval is often insufficient for domain-specific, relation-heavy questions.
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

## Contents

- `data/sample_finance_dataset.json`: Small finance dataset slice
- `framework/`: Reusable Python modules
- `notebooks/finance_graph_pipeline_tutorial.ipynb`: Notebook-style walkthrough

## Run The Framework Demo

```bash
python3 tutorials/finance_graph_pipeline/framework/demo.py
```

## Design Notes

- `FiboProfileAgent`: selects a constrained FIBO-like profile for each document
- `ExtractionAgent`: performs profile-conditioned extraction
- `EntityLinker`: normalizes aliases into canonical graph nodes
- `GraphQualityAnalyzer`: computes global graph quality and query-support path coverage
- `EvidenceSelector`: keeps only profile-matched, query-relevant evidence

This scaffold is designed so you can later replace the heuristic extraction and
answer generation with actual LLM calls, ontology resolution, and graph databases
without changing the minimal evaluation frame.
