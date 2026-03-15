# Experiment Setup

This document fixes the initial experiment setup for the tutorial companion so the stack is reproducible and the ontology effect is easy to inspect.

## Service Split

- `dozerdb`: canonical graph storage for extracted entities, relations, and graph-side diagnostics
- `runner`: isolated Python environment for OpenAI Agents SDK workflows, FinDER filtering, profile selection, extraction, evaluation, and metadata logging

The stack is intentionally split so graph persistence concerns do not leak into the Python runtime image.

## Storage Decisions

### Graph Store

Use DozerDB as the graph system of record. It is the right place for:

- canonical entity and relation materialization
- graph traversal and path-based support checks
- graph-quality diagnostics tied to nodes, edges, and subgraphs

### Metadata Store

Use SQLite as the primary metadata store for the experiment loop.

SQLite is the best default here because:

- the workload is operational rather than analytical
- writes are small and transactional
- schema evolution is simple
- it ships with Python and adds no extra service to the Compose stack
- it works well for run registries, profile decisions, evaluation records, and artifact pointers

### Artifact Format

Use JSON files for raw agent outputs and intermediate traces.

JSON should hold:

- raw extraction responses
- profile-agent rationales
- prompt inputs and model outputs
- graph-quality reports

Store the JSON file paths in SQLite instead of trying to normalize every raw payload into relational tables.

### DuckDB Position

Do not use DuckDB as the primary operational metadata store for the first experiment pass.

DuckDB is still useful later for:

- offline aggregate analysis across many runs
- ablation summaries
- metric pivot tables
- export into notebook-style analysis

The recommended pattern is:

1. DozerDB for the graph
2. SQLite for run metadata
3. JSON for raw artifacts
4. DuckDB only if aggregate analysis becomes heavy enough to justify it

## Agent Stack

Use the OpenAI Agents SDK in the `runner` container.

The immediate roles are:

- profile selection agent
- extraction agent
- normalization judge
- evaluation or critique agent

The `OPENAI_API_KEY` is expected to be passed in through `--env-file ../.env` when running Compose in this workspace.

If you are running the repository standalone after cloning it elsewhere, copy `.env.example` to `.env` and use `docker compose --env-file .env ...`.

## FinDER Scope

Use only the three FinDER categories where ontology effects are most visible:

- `Governance`
- `Financials`
- `Shareholder Return`

These are the right first categories because they best expose:

- role and committee structure normalization
- metric, period, and reporting-entity disambiguation
- corporate action and shareholder-return semantics

The filtering script writes:

- one combined parquet file for the top-3 subset
- one parquet file per selected category
- a `summary.json` file with schema and category counts

## FIBO Profile Agent Scope

Keep the FIBO profile agent thin.

It should do:

- ontology profile selection
- ontology-fit judgment for extracted entities and relations
- normalization decisions
- strict-versus-soft mapping policy output
- local-extension recommendation when FIBO coverage is insufficient

It should not do:

- full extraction
- large free-form reasoning over the whole document
- graph materialization
- end-to-end answer generation

## Initial FIBO Profiles

### Governance

- Entities: `LegalEntity`, `Organization`, `CorporateBoard`, `Committee`, `Person`, `Officer`, `Director`, `Role`, `Appointment`, `ControlRelationship`
- Relations: `has_board_member`, `holds_role`, `serves_on_committee`, `appointed_on`, `reports_to`, `controls`, `is_independent_director`, `is_chair_of`

### Financials

- Entities: `LegalEntity`, `BusinessSegment`, `FinancialReport`, `ReportingPeriod`, `MonetaryAmount`, `FinancialMetric`, `Currency`, `Guidance`, `ActualResult`
- Relations: `reports_metric`, `reported_for_period`, `reported_in_currency`, `belongs_to_segment`, `has_actual_value`, `has_guidance_value`, `compared_to_prior_period`, `attributed_to_entity`

### Shareholder Return

- Entities: `LegalEntity`, `EquityInstrument`, `ShareClass`, `Shareholder`, `Dividend`, `RepurchaseProgram`, `CorporateAction`, `CapitalAllocationPolicy`, `AnnouncementEvent`
- Relations: `issued_by`, `owned_by`, `pays_dividend`, `announced_repurchase`, `applies_to_share_class`, `returns_value_to_shareholders`, `authorized_on`, `affects_outstanding_shares`

## Metadata Schema

The SQLite schema is initialized from `sql/init_metadata.sql`.

Core tables:

- `documents`
- `experiment_runs`
- `profile_decisions`
- `graph_ingestion`
- `question_answers`
- `artifacts`

## Quick Start

Start the stack:

```bash
docker compose --env-file ../.env up -d dozerdb runner
```

Initialize SQLite metadata storage:

```bash
docker compose --env-file ../.env exec runner python scripts/init_metadata_db.py
```

Create the FinDER top-3 subset:

```bash
docker compose --env-file ../.env exec runner python scripts/filter_finder_top3.py
```

Run the minimal FinDER experiment:

```bash
docker compose --env-file ../.env exec runner python scripts/run_finder_experiment.py --per-category-limit 10 --persist-graph
```

Run the same pipeline against the curated manual gold subset:

```bash
docker compose --env-file ../.env exec runner python scripts/run_finder_experiment.py \
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

Build a proposal-facing summary with mean, 95% bootstrap CI, and category breakdown:

```bash
python3 scripts/build_proposal_metrics_report.py \
  --run-id <primary_run_id> \
  --manual-eval-run-id <manual_gold_run_id>
```

Inspect the generated summary:

```bash
cat data/processed/finder_top3/summary.json
```

Inspect experiment summaries:

```bash
ls data/experiment_runs
```

The runner-side pipeline now:

1. reads the filtered FinDER rows
2. records auxiliary `question_only` and `reference_only` answer baselines
3. performs profile selection
4. runs ontology-constrained extraction
5. materializes the graph and optionally writes it to DozerDB
6. writes run metadata to SQLite
7. stores per-example JSON artifacts and a run summary

The run summary keeps ontology-guided baselines in the main comparison table and
reports the two non-graph baselines separately as auxiliary checks.

The updated evaluation path also supports:

1. question-intent-specific query-support coverage
2. evidence-bundle graph answers built from selected triples
3. manual-gold extraction evaluation before any broader normalization work
