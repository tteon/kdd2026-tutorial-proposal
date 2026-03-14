# Finance Graph Pipeline Tutorial

This tutorial package demonstrates a three-stage graph pipeline for finance data:

1. Indexing
2. Graph quality analysis
3. Question answering with quality-aware routing

It is intentionally lightweight and heuristic-driven so you can inspect the flow
without depending on external LLM or graph infrastructure.

## Contents

- `data/sample_finance_dataset.json`: Small finance dataset slice
- `framework/`: Reusable Python modules
- `notebooks/finance_graph_pipeline_tutorial.ipynb`: Notebook-style walkthrough

## Run The Framework Demo

```bash
python3 tutorials/finance_graph_pipeline/framework/demo.py
```

## Design Notes

- The `FiboProfileAgent` selects a constrained FIBO-like profile for a document.
- The `ExtractionAgent` performs profile-conditioned extraction.
- The `GraphQualityAnalyzer` computes both global graph quality and query-support quality.
- The `QuestionAnsweringPipeline` uses quality metadata to decide which evidence is safe to use.

This scaffold is designed so you can later replace the heuristic extraction and
answer generation with actual agent, LLM, ontology, and graph database components.
