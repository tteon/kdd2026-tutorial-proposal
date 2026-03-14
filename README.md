# KDD 2026 Lecture-Style Tutorial Companion

This repository is a companion repository for a proposed KDD 2026 lecture-style tutorial on
ontology-grounded graph construction and quality-aware generation for domain-specific LLM systems.

The tutorial is motivated by a practical and research gap: many graph-grounded LLM systems discuss
retrieval and generation, but far fewer tutorials explain how ontology choices, extraction quality,
entity linking, graph quality diagnostics, and query-aware evidence selection interact in realistic
domain settings such as finance.

## Proposed Tutorial Focus

The lecture-style tutorial is centered on the following question:

How do ontology grounding and graph quality affect downstream generation in domain-specific LLM systems?

We use finance as a concrete case study and focus on:

1. Ontology-grounded graph construction
2. FIBO profile selection for domain slices
3. Entity and relation extraction under ontology constraints
4. Entity normalization and graph materialization
5. Graph quality diagnostics
6. Query-aware evidence selection for answer generation

## Intended Audience

- Researchers working on LLMs, knowledge graphs, GraphRAG, agent memory, or information extraction
- Practitioners building domain-specific LLM systems in high-stakes settings
- Data mining and KDD researchers interested in the interface between structured knowledge and generation

## Learning Objectives

By the end of the tutorial, participants should be able to:

1. Understand the major design patterns behind graph-grounded LLM systems
2. Explain why ontology quality, extraction quality, and graph quality should be evaluated separately
3. Recognize common failure modes in domain-specific graph construction pipelines
4. Design a quality-aware retrieval and answer generation workflow
5. Use finance as a case study for ontology-grounded evaluation with FIBO-style profiles

## Companion Materials In This Repository

### Proposal Materials

- `proposal/lecture_style_outline.md`
- `proposal/title_candidates.md`
- `proposal/abstract_draft.md`

### Runnable Companion Scaffold

- Notebook-style walkthrough:
  `tutorials/finance_graph_pipeline/notebooks/finance_graph_pipeline_tutorial.ipynb`
- Framework-style Python modules:
  `tutorials/finance_graph_pipeline/framework/`
- Small finance sample dataset:
  `tutorials/finance_graph_pipeline/data/sample_finance_dataset.json`

## Tutorial Outline

### Part 1. Motivation and Problem Setting

- Why domain-specific LLM systems need more than text-only retrieval
- Why ontology grounding matters for graph construction
- Why graph quality should be linked to downstream answer quality

### Part 2. Research Landscape

- GraphRAG, graph memory, and knowledge-engine design patterns
- Ontology-based information extraction and graph-grounded generation
- Domain-specific constraints in finance and other high-stakes settings

### Part 3. Pipeline Design

- Context ingestion
- FIBO profile selection
- Ontology-constrained extraction
- Entity linking and graph materialization
- Quality metadata propagation

### Part 4. Evaluation and Failure Analysis

- Ontology quality
- Extraction quality
- Graph quality
- Generation quality
- Ground-truth design and ablation settings

### Part 5. Finance Case Study

- Governance
- Financials
- Shareholder return
- FIBO subset reasoning

### Part 6. Open Research Questions

- Ontology drift and schema evolution
- Query-aware graph validation
- Confidence propagation across the pipeline
- Benchmarks for graph-grounded generation

## Quick Start For The Companion Scaffold

Run the framework demo:

```bash
python3 tutorials/finance_graph_pipeline/framework/demo.py
```

## Notes

- The runnable code is a companion scaffold, not the center of the lecture tutorial.
- The current implementation is heuristic-driven so the full workflow can be inspected without external model dependencies.
- The code is structured so actual LLM agents, ontology resolution, graph databases, and richer quality metrics can be swapped in later.
