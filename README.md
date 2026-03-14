# KDD 2026 Tutorial Proposal Assets

This repository contains a lightweight tutorial scaffold for a finance graph pipeline
that demonstrates:

1. Indexing with a FIBO profile selection step
2. Graph quality analysis
3. Question answering with quality-aware routing

The implementation is split into two views:

- Notebook-style walkthrough: `tutorials/finance_graph_pipeline/notebooks/finance_graph_pipeline_tutorial.ipynb`
- Framework-style Python modules: `tutorials/finance_graph_pipeline/framework/`

## Quick Start

Run the framework demo:

```bash
python3 tutorials/finance_graph_pipeline/framework/demo.py
```

## Tutorial Package

- `tutorials/finance_graph_pipeline/data/sample_finance_dataset.json`
- `tutorials/finance_graph_pipeline/framework/`
- `tutorials/finance_graph_pipeline/notebooks/`

## Notes

- The current implementation is heuristic-driven so the pipeline can be inspected without external model dependencies.
- The code is structured so actual LLM agents, ontology resolution, graph databases, and richer quality metrics can be swapped in later.
