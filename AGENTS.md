# AGENTS.md

## Purpose

This repository is a reproducible companion scaffold for a tutorial on
ontology-grounded graph construction and evaluation for domain-specific LLM
systems.

All work in this repository must preserve two priorities:

1. experimental interpretability
2. reproducibility of reported results

## Repository-Level Rules

### 1. Keep The Causal Story Minimal

Every code change should support the same minimal chain:

`ontology/profile selection -> extraction -> linking/graph construction -> graph quality -> answer quality`

Do not add large subsystems unless they are necessary to make this chain more
measurable or more reproducible.

### 2. Reproducibility Comes Before Cleverness

Prefer:

- simple Python over hidden orchestration
- explicit config fields over implicit defaults
- serialized JSON/CSV artifacts over opaque in-memory state
- deterministic or seeded sampling where possible
- balanced evaluation slices over uncontrolled full-dataset comparisons

Avoid:

- undocumented one-off experiment scripts
- hidden local assumptions
- changing prompts, budgets, or evaluation rules between baselines without
  documenting the reason

### 3. Preserve Fair Baseline Comparison

For reported comparisons:

- keep the answer model fixed across baselines
- keep the answer instruction fixed across baselines
- keep the answer output style fixed across baselines
- keep the answer context budget fixed across baselines
- only change the evidence source unless the comparison is explicitly about the
  answer layer

If a change introduces a new confound, document it explicitly.

### 4. Treat Generated Results As Local Artifacts

Generated outputs under `data/` and `exports/` are local experiment artifacts.

They should:

- be reproducible from code and documented commands
- not be the only source of truth for the repo design
- not be required for CI

Static proposal-facing assets under `proposal/figures/` are allowed when they
are regenerated from tracked scripts and tracked local summaries.

### 5. Keep Experiment Reporting Structured

Any new reported experiment should preserve:

- run id
- dataset scope
- baseline definitions
- model and answer mode
- metric definitions
- category breakdown
- confidence interval method when reported

If a metric changes meaning, update the docs and reporting scripts together.

## Code Style

- Keep modules inspectable and small.
- Prefer dataclasses, typed dictionaries, and explicit return payloads over
  deeply nested ad hoc structures.
- Keep prompts and evaluation policies near the code paths that use them.
- Use ASCII unless the file already needs non-ASCII.
- Keep comments short and only where the code is not self-evident.

## Experiment Design Rules

- Main reported slices should be category-balanced.
- Main reported runs should be reproducible from documented commands.
- Auxiliary baselines may be retained for interpretation, but the primary
  narrative should stay centered on ontology-guided graph comparisons.
- Manual-gold evaluation should be used when coarse proxy metrics are known to
  be misleading.

## Documentation Rules

When behavior changes, update the relevant docs in the same change set.

Current documentation layout:

- `docs/setup/`
- `docs/reviews/`
- `docs/analysis/`
- `proposal/`

At minimum, update:

- `README.md` for repo-level flow changes
- `docs/reviews/` for experiment-logic changes
- `scripts/README.md` for operator workflow changes

## CI Expectations

CI should remain lightweight and reproducible.

Preferred checks:

- Python syntax compilation
- sample-dataset evaluator run
- CLI entrypoint help/smoke checks

Avoid CI jobs that require local experiment artifacts, private credentials, or
long-running OpenAI-backed runs.

## Release And Push Hygiene

Before pushing:

1. remove transient cache directories such as `__pycache__/`
2. run lightweight verification
3. check that generated local artifacts are still excluded as intended
4. verify that new docs point to the correct current paths

## Local Domain Guidance

The finance tutorial package has stricter local instructions in:

- `tutorials/finance_graph_pipeline/AGENTS.md`

Treat the local finance-pipeline guidance as an extension of this file, not a
replacement.
