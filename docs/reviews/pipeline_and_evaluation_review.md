# Pipeline And Evaluation Review

This note is the current review draft for the FinDER finance graph pipeline.
It is meant to confirm the experimental claim, the baseline setup, and the
evaluation logic before we interpret larger runs.

## Objective

We want to test one minimal claim:

Ontology-guided graph construction changes graph quality, and that change can
affect downstream answer quality.

The working dataset scope is the FinDER top-3 slice:

- Governance
- Financials
- Shareholder Return

## Pipeline

The current minimal pipeline is:

1. Document ingestion from FinDER references
2. FIBO-style profile selection
3. Ontology-constrained extraction
4. Entity linking and graph materialization
5. Graph quality analysis
6. Question-intent-specific query-support analysis
7. Quality-aware evidence selection
8. Answer generation

The answer generation layer now follows one fixed design:

- same answer model across baselines
- same answer instruction across baselines
- same output style across baselines
- same context budget across baselines
- only the provided context changes

## Baselines

The comparison set is:

- `question_only_baseline`
- `reference_only_baseline`
- `graph_without_ontology_constraints`
- `graph_with_profile_selection_only`
- `graph_with_profile_plus_constrained_extraction_and_linking`
- `full_minimal_pipeline_with_quality_aware_evidence_selection`

There is also one supplemental mode:

- `--answer-mode heuristic_synthesis`

This should be treated as an answer-layer comparison only. It is useful for
showing the effect of switching from legacy code-based answer synthesis to the
shared answer agent while keeping the upstream graph pipeline fixed.

## Answering Policy

The main answer comparison now uses a shared answer agent.

This is the preferred design for the main experiment because it keeps the
answerer fixed and changes only the provided context.

The prompt is fixed and enforces:

- answer only from provided context
- do not use outside knowledge
- answer conservatively when support is incomplete
- state what is grounded versus missing

The provided context is serialized as structured text.

`question_only`
- question
- explicit statement that no supporting evidence is available

`reference_only`
- question
- reference-only context section
- top-ranked reference sentences under the shared budget

`graph`
- question
- intent id
- entities
- triples
- provenance snippets
- missing slots

The shared answer context is truncated to a single fixed budget.

## Main Metrics

The four primary metrics remain:

- `profile_selection_accuracy`
- `ontology_constrained_extraction_f1`
- `query_support_path_coverage`
- `answer_quality_delta`

`answer_quality_delta` should be interpreted across the baseline matrix, not as
one single comparison only.

## Secondary Metrics

The current supporting metrics are:

- `required_answer_slot_coverage_manual`
- `preferred_answer_evidence_hit_rate`
- `relation_schema_conformance_rate`
- `fallback_error_rate`

Reports should include:

- overall mean
- 95 percent bootstrap CI
- category breakdown

## Evaluation Sets

Current target evaluation sizes:

- primary: `50/category`
- supplemental answer-layer comparison: `50/category` with `--answer-mode heuristic_synthesis`
- robustness: `90/category`
- manual gold: `10-15/category`

The manual gold subset is used to check whether extraction and answer support
are aligned with the intended question slots before scaling interpretation.

## Current Review Questions

Please review these points specifically:

1. Is the baseline set complete enough for the tutorial claim?
2. Is the answer-agent design fair enough across `question`, `reference`, and
   `graph` contexts?
3. Is the structured graph context format sufficient, or should we simplify or
   add fields?
4. Are the four primary metrics plus four supporting metrics enough for the
   proposal table?
5. Should the proposal highlight the legacy heuristic answer mode at all, or
   leave it out of the main narrative?
