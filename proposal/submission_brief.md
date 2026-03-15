# Submission Brief

## One-Line Claim

Ontology-guided graph construction improves downstream answering over naive
graph construction, but strong reference-text baselines still expose the
remaining grounding gap.

## Problem

Many GraphRAG-style systems discuss retrieval and generation, but fewer show
how ontology choice, extraction quality, graph quality, and answer quality are
connected in a domain setting.

This tutorial uses finance to make that chain concrete.

## Scope

Dataset:

- FinDER top-3 categories only
- Governance
- Financials
- Shareholder Return

Ontology setup:

- FIBO-style profile selection
- ontology-constrained extraction
- entity linking and graph materialization

## Main Pipeline

1. Select one domain profile.
2. Extract typed entities and relations under ontology constraints.
3. Link entities and build the graph.
4. Measure graph quality and query-support coverage.
5. Generate answers with one shared answer agent.
6. Compare all baselines against FinDER ground-truth answers.

## Fairness Rule

The answerer is fixed across all baselines.

Only the context changes:

- question only
- reference text only
- graph evidence bundle

This keeps the answer layer from becoming the main confound.

## Main Baselines

- `question_only_baseline`
- `reference_only_baseline`
- `graph_without_ontology_constraints`
- `graph_with_profile_selection_only`
- `graph_with_profile_plus_constrained_extraction_and_linking`
- `full_minimal_pipeline_with_quality_aware_evidence_selection`

## Main Metrics

- `profile_selection_accuracy`
- `ontology_constrained_extraction_f1`
- `query_support_path_coverage`
- `answer_quality_delta`

Supporting metrics:

- `required_answer_slot_coverage_manual`
- `preferred_answer_evidence_hit_rate`
- `relation_schema_conformance_rate`
- `fallback_error_rate`

## Main Result From The Current 50/Category Run

- `reference_only = 0.360`
- `graph_without_ontology_constraints = 0.175`
- `graph_with_profile_selection_only = 0.175`
- `graph_with_profile_plus_constrained_extraction_and_linking = 0.318`
- `full = 0.321`

Interpretation:

- ontology-guided graph construction clearly beats naive graph construction
- profile selection alone is not enough
- constrained extraction and linking create most of the gain
- the current graph pipeline still trails the strong reference-text baseline

## Category-Level Reading

- `Financials` is strongest: graph nearly matches `reference_only`
- `Governance` improves over naive graph baselines but still misses role- and
  committee-specific grounding
- `Shareholder Return` improves substantially but still struggles with
  authorization, remaining capacity, and policy-vs-action distinctions

## Safe Proposal Takeaway

The tutorial should claim that ontology-guided graph construction changes graph
quality and materially improves downstream answering relative to naive graph
construction.

It should not claim that the current graph pipeline already beats a strong
reference-text baseline.

## Reusable Compressed Sentences

Motivation:

This tutorial asks how ontology grounding changes graph quality and downstream
generation in domain-specific LLM systems.

Method:

We compare question-only, reference-only, and ontology-guided graph baselines
under one shared answer agent so that only the evidence source changes.

Result:

In a FinDER finance case study, ontology-guided graph construction improves
answer quality sharply over naive graph construction, while strong
reference-text baselines remain competitive.

Takeaway:

The tutorial teaches not only how to build ontology-grounded graphs, but also
how to evaluate when graph quality does and does not transfer to downstream
answer quality.
