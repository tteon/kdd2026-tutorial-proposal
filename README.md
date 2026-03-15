# KDD 2026 Lecture-Style Tutorial Companion

This repository is a companion repository for a proposed KDD 2026 lecture-style tutorial on
ontology-grounded graph construction and quality-aware generation for domain-specific LLM systems.

The tutorial is motivated by a practical and research gap: many graph-grounded LLM systems discuss
retrieval and generation, but far fewer explain how ontology choices, extraction quality, entity
linking, graph quality, and downstream answer quality interact in realistic domain settings such as
finance.

## Minimal Core Claim

The tutorial focuses on one minimal claim:

**Downstream answer quality in domain-specific LLM systems depends on whether the system constructs
query-supporting graphs with sufficient ontology alignment, extraction quality, and graph quality.**

To keep the claim testable, the companion implementation keeps only the minimum pipeline needed to
support that argument:

1. Profile selection
2. Ontology-constrained extraction
3. Entity linking
4. Graph construction
5. Graph quality analysis
6. Quality-aware evidence selection
7. Answer generation

We intentionally do **not** make question routing, multi-agent orchestration, UI, or deployment a
core part of the tutorial argument.

## Proposed Tutorial Focus

The lecture-style tutorial is centered on the following question:

How do ontology grounding and graph quality affect downstream generation in domain-specific LLM systems?

We use finance as a concrete case study and focus on:

1. Ontology-grounded graph construction
2. FIBO profile selection for domain slices
3. Entity and relation extraction under ontology constraints
4. Entity normalization and graph materialization
5. Graph quality diagnostics
6. Quality-aware evidence selection for answer generation

## What Must Be Shown

The tutorial is organized around four measurable quantities:

| Quantity | Why it is needed |
|---|---|
| Profile Selection Accuracy | Shows whether ontology/profile routing is correct |
| Ontology-Constrained Extraction F1 | Shows whether the selected ontology improves structured extraction |
| Query-Support Path Coverage | Shows whether the graph contains the question-intent-specific subgraph needed to answer |
| Answer Quality Delta | Shows whether graph quality transfers to downstream answer quality |

These four quantities are enough to support the end-to-end argument without introducing unnecessary
system complexity.

For downstream answer comparison, the scaffold now uses one shared answer agent
with a fixed prompt, fixed model, and shared context budget across
`question_only`, `reference_only`, and graph baselines. The legacy code-based
answer synthesis path remains available only as a supplemental mode.

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

## Reproducible Experiment Stack

The repository now includes a reproducible Docker Compose stack with separate services for:

- DozerDB graph storage
- a Python runner environment for OpenAI Agents SDK workflows and FinDER preprocessing

Key files:

- `AGENTS.md`
- `docker-compose.yml`
- `docker/python/Dockerfile`
- `.github/workflows/ci.yml`
- `docs/README.md`
- `docs/setup/experiment_setup.md`
- `docs/reviews/pre_submission_pipeline_experiment_review_20260315.md`
- `scripts/README.md`
- `scripts/filter_finder_top3.py`
- `scripts/init_metadata_db.py`
- `scripts/run_finder_experiment.py`
- `scripts/evaluate_manual_gold_subset.py`
- `scripts/build_proposal_metrics_report.py`
- `scripts/build_submission_assets.py`
- `tutorials/finance_graph_pipeline/data/finder_manual_gold_subset.json`

Generated run outputs under `data/` and `exports/` are kept local and are intentionally excluded from the repository push.

Documentation layout:

- `docs/setup/`
  Experiment setup and stack notes
- `docs/reviews/`
  Review notes and submission checks
- `docs/analysis/`
  Failure analysis
- `proposal/`
  Submission-facing drafts

Reproducibility conventions:

- `AGENTS.md`
  Repo-wide implementation and experiment rules
- `tutorials/finance_graph_pipeline/AGENTS.md`
  Finance-pipeline-specific experimental guidance
- `.github/workflows/ci.yml`
  Lightweight syntax and sample-evaluation checks

## Tutorial Outline

### Part 1. Motivation and Problem Setting

- Why domain-specific LLM systems need more than text-only retrieval
- Why ontology grounding matters for graph construction
- Why graph quality should be linked to downstream answer quality

### Part 2. Research Landscape

- GraphRAG, graph memory, and knowledge-engine design patterns
- Ontology-based information extraction and graph-grounded generation
- Domain-specific constraints in finance and other high-stakes settings

### Part 3. Minimal Pipeline Design

- Context ingestion
- FIBO profile selection
- Ontology-constrained extraction
- Entity linking and graph materialization
- Minimal quality metadata propagation

### Part 4. Evaluation and Failure Analysis

- Profile selection accuracy
- Ontology-constrained extraction quality
- Graph quality and query-support paths
- Answer quality under baseline and ablation settings

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
- The current implementation supports both heuristic mode and OpenAI Agents SDK mode for profile selection, extraction, and answering.
- The implementation deliberately excludes non-essential components so the tutorial can focus on the minimal causal chain from ontology choice to answer quality.
