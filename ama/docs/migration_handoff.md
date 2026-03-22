# AMA Migration Handoff

## Current state

- Indexing baseline is available from `artifacts/rdf_vs_lpg_balanced_pilot30`.
- Debate/E2E is run as **example-scoped**, **reference-bound** GraphRAG.
- Retrieval contract is documented in:
  - `AGENTS.md`
  - `EXPERIMENT_SETUP.md`
  - `Documentation.md`
- Current active issue stream is `.beads` issue `ama-cqu`.

## What is already working

- Indexing/materialization over all 8 FinDER categories on the pilot slice.
- Debate traces log to Opik Cloud.
- Debate metadata includes:
  - `pipeline_family`
  - `run_kind`
  - `trace_display_name`
  - `retrieval_scope_mode=example_scoped`
  - `reference_bound=true`
  - `source_experiment`
  - `scoped_example_id`
- Search-A retrieval variants are implemented:
  - `search_a_v1`
  - `search_a_v3`
  - `search_a_v4`
  - `search_a_v5`

## Current recommended variants

- `Risk` → `search_a_v5`
- `Legal` → `search_a_v5`
- `Footnotes` → `search_a_v5`
- `Financials` → `search_a_v5`
- `Governance` → `search_a_v5` for now, but still judge-heavy
- `Accounting` → `search_a_v3`

## Remaining issues

- `entity anchor sanity`
  - company/ticker anchoring is still imperfect
  - most visible in Governance and other support-sparse cases
- `output robustness`
  - synthesis wrapper bug was patched
  - rerun validation started, but final expanded batch was intentionally paused
  - if resuming elsewhere, re-run a 1-example smoke first

## Important code paths

- Debate orchestration: `debate/debate_pool.py`
- Debate runner: `experiments/run_debate.py`
- Opik tracing: `tracing/tracing.py`
- Indexing runner: `experiments/run_indexing.py`
- Example chain inspector: `experiments/inspect_example_chain.py`

## Validation already done

- `py_compile` passed for:
  - `tracing/tracing.py`
  - `debate/debate_pool.py`
  - `experiments/run_debate.py`
  - `experiments/run_indexing.py`

## Restart sequence in a new environment

1. Restore env vars and local services.
2. Verify DozerDB / Neo4j connectivity.
3. Run a 1-example debate smoke:
   - category of choice
   - `source_experiment=rdf_vs_lpg_balanced_pilot30`
4. Confirm Opik trace metadata contains:
   - `retrieval_scope_mode=example_scoped`
   - `reference_bound=true`
   - `source_experiment`
   - `scoped_example_id`
5. Restart the larger balanced batch only after the smoke passes.

## Suggested first smoke command

```bash
uv run python -m experiments.run_debate \
  --dataset-id Linq-AI-Research/FinDER \
  --split train \
  --sample-size 1 \
  --experiment-name debate_opik_scope_check \
  --category Governance \
  --source-experiment rdf_vs_lpg_balanced_pilot30 \
  --retrieval-variant search_a_v5
```

## Suggested expanded batch

- `Governance` → `search_a_v5`
- `Financials` → `search_a_v5`
- `Risk` → `search_a_v5`
- `Legal` → `search_a_v5`
- `Accounting` → `search_a_v3`
- `Footnotes` → `search_a_v5`
- start with `sample_size=20`

## Handoff note

The current goal is not framework generalization. The goal is a credible,
category-balanced, example-scoped GraphRAG experiment over the same mapped
FinDER references, with RDF/LPG divergence attributable to indexing and
materialization rather than drifting retrieval scope.
