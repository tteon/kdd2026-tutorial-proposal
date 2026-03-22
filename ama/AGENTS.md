# AGENTS.md

## Purpose

This repository supports an observability-first experiment for domain-specific graph construction.

The detailed experiment specification lives in [`EXPERIMENT_SETUP.md`](EXPERIMENT_SETUP.md).
Read that file before implementing experiment logic, evaluation logic, or later search/generation debate logic.

## What Belongs Here

`AGENTS.md` should remain lightweight and agent-oriented.

Use it for:

- mandatory execution rules
- current phase boundaries and progress
- pointers to the detailed experiment setup
- project workflow instructions

Do **not** keep the full experiment specification only in this file.

## Current Phase: Indexing Evaluation

Phase progress (as of 2026-03-22):

- [x] spec and artifact definitions
- [x] Opik trace helpers (cloud + local fallback both available)
- [x] smoke run verified (6 examples x 2 reps = 12/12 completed, gpt-4o)
- [x] seed gold curated (6 examples, 39 entities — pipeline-derived, diagnostic only)
- [x] dual-axis evaluation: gold-based P/R/F1 + FIBO conformance (no gold needed)
- [x] Opik Cloud connection verified (manual debug trace sent successfully)
- [x] Opik Prompt auto-versioning (extraction/linking/relation per ontology_mode)
- [x] Opik BaseMetric evaluation framework (FIBOConformance, SelfConsistency, GoldExtraction, GoldLinking)
- [x] Opik batch ablation evaluation (eval_ablation.py — all experiments with experiment_config)
- [x] balanced divergent-indexed-view artifacts exist from shared raw evidence
- [x] ontology ablation artifacts exist (`none` vs `static` vs `dynamic`)
- [x] dynamic FIBO OWL loading (real FIBO annotations via rdflib, 11 FIBO + 13 AMA concepts)
- [x] DozerDB materialization truth normalized in code
  - single instance, `finderrdf` + `finderlpg`
  - do not reintroduce alternate storage truths such as Oxigraph
- [ ] live docker-backed DozerDB materialization verification rerun pending
  - rerun `rdf_vs_lpg_balanced --materialize` against `finderrdf` and `finderlpg`
  - only after that should debate/search be treated as unblocked
- [ ] debate pool end-to-end test
  - code exists, but should remain blocked on the restored DozerDB materialization rerun

The current phase is **indexing and observability**, not full QA benchmarking.

Do not optimize downstream answer quality before extraction, linking, canonicalization, and materialization behavior are inspectable.

## Current FinDER Task Framing

For the current experiment phase, keep the role of each FinDER field explicit:

- `references` = the **primary indexing surface**
  - use this as the main input for entity extraction, normalization, and linking
- `text` = the **selection signal**
  - use this mainly to help choose the task-relevant ontology slice
- `answer` = the **weak evaluation / alignment signal**
  - use this only as a downstream sanity-check or auxiliary inspection signal

This means the current core task is:

- **ontology-guided indexing over `references`**

not:

- answer generation optimization
- full QA benchmarking
- broad multi-agent reasoning over the whole example

Current pipeline framing:

- `text + references`
  → ontology slice selection
  → ontology-derived guidance pack construction
  → entity extraction over `references`
  → entity linking
  → RDF / LPG materialization
  → optional answer-alignment inspection

Current prompt stack:

- `ontology selection prompt`
  - decides `none | static | dynamic` guidance content in effect for the example
- `entity extraction prompt`
  - uses the selected guidance pack to extract mentions from `references`
- `entity linking prompt`
  - uses the selected guidance pack to constrain FIBO concept linking
- `relation extraction prompt`
  - uses the same guidance pack to extract semantic relations between mentions

Prompt previews and hashes for these prompts should remain visible in traces and artifacts.

For GraphRAG runs built on top of these indexed artifacts, keep retrieval scope explicit:

- retrieval is **example-scoped**, not open-ended corpus search
- `question` stays bound to its mapped FinDER `references`
- RDF/LPG agents read only the graph views materialized from that same example-bound evidence
- representation differences should come from divergent ontology-guided indexing and materialization, not from drifting source slices

Trace and manifest metadata should make this visible with fields such as:

- `retrieval_scope_mode=example_scoped`
- `reference_bound=true`
- `source_experiment`
- `example_id`

If a proposal shifts the center of gravity away from `references`-based indexing, treat it as out of scope for the current phase unless explicitly approved.

## DozerDB Runtime Rule

For graph materialization and later debate/search runs, treat DozerDB as:

- one DozerDB instance
- one shared `NEO4J_URI`
- two logical databases inside that instance
  - RDF database: `finderrdf`
  - LPG database: `finderlpg`

Use:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE_RDF=finderrdf`
- `NEO4J_DATABASE_LPG=finderlpg`

Do **not** assume:

- separate RDF/LPG containers
- separate `NEO4J_RDF_URI` / `NEO4J_LPG_URI`
- underscore-bearing database names such as `ama_rdf` or `ama_lpg`

When implementing agents or graph tools:

- `RDFAgent` must query `finderrdf`
- `LPGAgent` must query `finderlpg`
- trace metadata should preserve the target database name
- do not reintroduce alternate storage truths such as Oxigraph or split RDF/LPG Bolt endpoints unless the experiment spec is explicitly changed first

## Research-First Coding Rule

This repository is **research-first and experiment-first**, not production-first.

When implementing here, prioritize:

- readability
- inspectability
- reproducibility
- traceability
- presentation-ready clarity

The user must be able to read the code, explain the pipeline in a paper or talk, and inspect intermediate outputs without reverse-engineering hidden framework behavior.

## Required Reading

Before making experiment-related changes, consult:

1. [`EXPERIMENT_SETUP.md`](EXPERIMENT_SETUP.md) — full experiment spec
2. [`CODEBASE_MAP.md`](CODEBASE_MAP.md) — reading order and runtime path
3. [`context-ONTOLOGY_GUIDE.md`](context-ONTOLOGY_GUIDE.md) — FIBO best practices
4. [`docs/ama_master_report.md`](docs/ama_master_report.md) — research narrative and findings

For long-running Codex execution in this repository, also use:

5. [`Prompt.md`](Prompt.md) — frozen target for the current run
6. [`Plan.md`](Plan.md) — milestone and validation sequence
7. [`Implement.md`](Implement.md) — execution runbook
8. [`Documentation.md`](Documentation.md) — live status / audit log

## Non-Negotiable Experiment Rules

- Compare RDF and LPG on the same example slice whenever the comparison is intended.
- Keep prompts, evaluator logic, and trace schema aligned unless a minimal representation-specific change is required and documented.
- Prefer smaller, more inspectable designs over broader but opaque systems.
- Treat hybrid as an explicit design choice, not an undefined union of outputs.
- Treat search/generation debate as a later phase built on top of stable indexed artifacts, not a substitute for weak indexing.
- If debate is implemented, use dual tracing:
  OpenAI Agents SDK tracing for agent-internal execution, and Opik tracing for experiment-level spans, handoff metadata, and agent-graph visibility.
- Do not treat a single coarse `execute_strategy` span as sufficient debate observability.
- If retrieval is introduced, prefer `Search-A`: question entity/concept parsing, canonical mapping, and constrained query templates.
- Defer vector-first retrieval and open-ended `text2cypher` until after Search-A is stable and measurable.
- Prefer explicit pipelines over clever abstraction.
- Prefer intermediate artifacts over hidden state.
- Prefer code that is easy to present over code that is merely compact.
- Treat ontology selection as part of the indexing pipeline, not as an external preprocessing convenience.
- Do not treat multi-agent ontology reasoning as the current main task; it is future work until the indexing pipeline is stable.

## Done / Not Done Rule

An implementation task in this repository is **not done** if any of the following remain true:

- RDF/LPG representation differences are only described in prose but not visible in runtime code
- ontology guidance is mentioned but not wired into prompt construction, artifacts, or traces
- a real ontology source is not consulted and the fallback/hardcoded path is not explicitly disclosed
- artifact or trace fields needed to inspect the change are missing
- DozerDB routing assumptions differ from the documented single-instance + logical-database rule
- placeholders remain in the critical path without being called out explicitly

An implementation task is **done only if** the agent reports:

- what was actually implemented
- what remains placeholder or deferred
- which trace/artifact fields now expose the behavior
- what was verified locally

Use explicit self-audit language rather than vague “implemented” claims.

## When In Doubt

- Choose the smaller experiment.
- Choose the design with clearer traces.
- Choose the artifact that better explains failure without rerunning.
- Record asymmetries explicitly.
- Choose the implementation that a researcher can explain on a slide.

## Directory and File Expectations

Current project structure:

- `framework/` — extraction, linking, materialization, evaluation
  - `schema.py` — core dataclasses (ExampleArtifact, ExperimentConfig, etc.)
  - `pipeline.py` — linear extraction pipeline (7 stages)
  - `ontology.py` — static FIBO pack selection
  - `prompt_rules.py` — ontology-to-prompt translation
  - `llm_models.py` — Pydantic models for OpenAI structured output
  - `llm_client.py` — thin OpenAI wrapper
  - `evaluation.py` — gold-based eval + FIBO conformance eval
  - `analysis.py` — comparison and summary helpers
  - `graph_store.py` — DozerDB/Neo4j write path
  - `rdf_export.py` — Turtle export for RDF artifacts
  - `neo4j_safe.py` — Neo4j-safe label/type sanitization
  - `loader.py` — FinDER dataset loading + balanced sampling
- `experiments/` — entry points and runners
  - `run_indexing.py` — unified indexing runner (replaces smoke + balanced)
  - `eval_smoke.py` — dual-axis evaluation + optional Opik evaluate()
  - `eval_ablation.py` — batch ablation evaluation across all experiments
  - `run_debate.py` — debate pool runner (later phase)
  - `verify_traces.py` — trace completeness checker
  - `render_prompt_rules.py` — prompt inspection
  - `category_entity_overlap.py` — entity overlap analysis
  - `finder_category_signal_analysis.py` — FinDER lexical analysis
- `tracing/` — Opik + local trace adapters
  - `tracing.py` — OpikTracer (cloud), LocalTraceRecorder (JSON fallback)
- `debate/` — multi-agent debate pool (later phase)
  - `debate_pool.py` — orchestrator
  - `rdf_agent.py`, `lpg_agent.py`, `synthesis_agent.py` — OpenAI Agents SDK
  - `graph_tools.py` — DozerDB query tools for agents
  - `schemas.py` — debate I/O contracts

Current debate architecture rule:

- `RDFAgent`
- `LPGAgent`
- deterministic `RouterPolicy` inside `DebatePool`
- `SynthesisAgent`

Do not silently introduce a separate router LLM agent in the current phase.
If router-agent experiments are added later, treat them as a new experimental condition.
- `manual_gold/` — manual evaluation subset
  - `gold_schema.py` — GoldEntity, GoldExample
  - `seed_gold.json` — diagnostic gold (pipeline-derived, not publication-grade)
- `artifacts/` — local JSON outputs (gitignored)
- `docs/` — research documentation and analysis

Do not create extra layers unless needed.

## Evaluation Model

Two evaluation axes exist:

1. **Gold-based** (Axis 1) — P/R/F1 for extraction, accuracy for linking.
   Current seed gold is pipeline-derived and circular. Use for debugging only.
   Publication-grade gold requires independent human annotation.

2. **FIBO conformance** (Axis 2) — No gold needed. Measures how well pipeline output follows FIBO Ontology Guide best practices:
   entity type naming (UpperCamelCase, singular), concept ID format (FIBO prefix, known domain),
   label uniqueness, RDF triple hygiene (predicate lowerCamelCase, typed subjects/objects).
   Grounded in `context-ONTOLOGY_GUIDE.md`.

3. **Opik evaluate() framework** (Axis 3) — Structured experiment comparison in Opik dashboard.
   Four BaseMetric subclasses: FIBOConformanceMetric, SelfConsistencyMetric, GoldExtractionMetric, GoldLinkingMetric.
   Experiment-level scores: macro_fibo_overall, macro_gold_f1.
   Usage: `eval_smoke.py --opik` (single experiment), `eval_ablation.py` (batch ablation comparison).
   Each experiment gets distinct `experiment_config` (ontology_mode, model, sample_size) for side-by-side comparison.

Smoke run findings (gpt-4o, 6 examples):
- RDF: higher precision (0.93), better FIBO concept ID conformance (1.00 across all dimensions)
- LPG: higher recall in simple categories, cleaner graph structure (no untyped subjects)
- Overall conformance: RDF 0.93, LPG 0.90

## What To Do Now vs Later

### Do Now

- improve ontology slice selection from `text + references`
- improve ontology-derived guidance packs
- improve extraction/linking over `references`
- compare RDF and LPG materialization on the same indexed artifacts
- improve traces so selection, extraction, linking, and materialization failures are attributable

### Do Later

- end-to-end answer optimization
- retrieval-heavy QA experiments
- full multi-agent ontology reasoning pipelines
- open-ended debate/search generation as the main comparison target

Use this split to avoid drifting away from the original observability-first indexing design.

## Experiment Naming

Use stable experiment names:

- `rdf_vs_lpg` — default experiment name
- `rdf_vs_lpg_balanced_*` — ablation variants (none/static/dynamic/all_packs/rule_only)
- `debate_rdf_vs_lpg` — debate pool runs

Each run records: version, date, model, prompt_template_id.

## Success Criteria

This phase is successful if we can clearly inspect:

- what each representation extracted
- what each representation linked correctly or incorrectly
- where canonicalization broke down
- whether materialization succeeded
- whether traces are complete enough to explain failures without rerunning blindly

Success does **not** require one representation to win overall.
Success requires interpretable comparison.

## Failure Interpretation

Failures are useful if they remain attributable.

The implementation should make it possible to explain failures in terms of:

- missed entity extraction
- over-extraction
- normalization mismatch
- ambiguous linking
- null linking
- representation-specific materialization mismatch
- schema or type mismatch (now measurable via FIBO conformance)
- prompt mismatch
- fallback behavior
- evaluator weakness
- trace incompleteness

Do not optimize only for better aggregate scores.
Optimize for inspectable, explainable behavior.

## Agent Working Rules

When implementing from this directory:

1. ~~start with spec and artifact definitions~~ (done)
2. ~~implement trace helpers before large experiments~~ (done, cloud + local fallback)
3. ~~verify trace completeness on a smoke slice~~ (done, 12/12)
4. ~~add evaluation before scaling~~ (done: gold-based + FIBO conformance)
5. run balanced shared-evidence / divergent-indexed-view comparisons after the above are stable (next)
6. run ontology ablation (`none` vs `static` vs `dynamic`) on the same balanced slice
7. DozerDB materialization + graph-level inspection (requires docker)
8. debate pool testing (requires populated graph stores)

When uncertain, choose the smaller, more inspectable design.

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:d4f96305 -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Hierarchy Rule

For this repository, experiment-scale work should use **hierarchical beads** whenever possible.

Use the hierarchy intentionally:

- epic / parent bead
  - experiment stream, architecture change, or phase-level objective
- child bead
  - concrete executable task under that stream
- sub-task bead
  - bounded follow-up, fix, or implementation slice under the child

This means:

- do **not** create flat standalone issues for work that clearly belongs to an existing experiment stream
- do **not** open architecture-changing tasks without attaching them to an epic or parent bead
- do **not** close child work without recording what changed, what remains, and whether parent assumptions shifted

Use parent/child structure especially for:

- indexing pipeline work
- ontology-guidance work
- storage/materialization work
- debate/tracing work
- evaluation and ablation work

Trivial one-off housekeeping can remain flat, but experiment logic should be nested.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- END BEADS INTEGRATION -->
