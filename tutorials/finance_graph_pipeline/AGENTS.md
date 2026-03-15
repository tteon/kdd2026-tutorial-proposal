# AGENTS.md

This file extends the repo-level rules in `../../AGENTS.md`.
When the two conflict, preserve the stricter rule for experimental
interpretability and reproducibility.

## Purpose

This tutorial package exists to support one minimal tutorial claim:

**In domain-specific LLM systems, downstream answer quality depends on whether the system constructs query-supporting graphs with sufficient ontology alignment, extraction quality, and graph quality.**

All implementation work in this directory must serve that claim directly.

## Important Principle

All tutorial sections must be represented in the implementation, but only through the minimum functional components needed to support the tutorial's causal claim.

Do **not** create separate large subsystems for every lecture section.

Lecture breadth belongs in the tutorial narrative.
Implementation should stay minimal and causal.

## Tutorial Sections That Must Be Covered

The tutorial content covers:

- ontology
- indexing
- graph quality
- evaluation
- answering
- finance case study

These sections must all be represented, but only through the smallest implementation needed.

## Minimal Mapping From Sections To Code

- `ontology` -> profile selection + ontology-constrained extraction
- `indexing` -> entity linking + graph construction
- `graph quality` -> schema checks + duplicate checks + disconnected graph checks + query-support path checks
- `evaluation` -> metric computation + baseline comparison
- `answering` -> quality-aware evidence selection + answer generation
- `finance case study` -> sample dataset + query templates

## Minimal Pipeline

Keep only the following stages:

1. Profile selection
2. Ontology-constrained extraction
3. Entity linking / normalization
4. Graph construction
5. Graph quality analysis
6. Quality-aware evidence selection
7. Answer generation

This is the full intended implementation scope for the tutorial companion.

## Out Of Scope

Do **not** add any of the following unless explicitly requested:

- question router agents
- multi-agent orchestration
- dynamic ontology extension
- UI / dashboard
- deployment / serving logic
- graph database optimization
- complex temporal reasoning extensions
- autonomous planning loops
- extra metadata systems beyond what is required for evaluation
- benchmark breadth beyond the minimal finance case study

If a component does not help demonstrate the minimal causal chain, do not implement it.

## What The Implementation Must Prove

The implementation must make the following four quantities measurable:

1. `profile_selection_accuracy`
2. `ontology_constrained_extraction_f1`
3. `query_support_path_coverage`
4. `answer_quality_delta`

These are the only primary metrics that matter for the tutorial claim.

Supporting metrics that should remain available when possible:

- `required_answer_slot_coverage_manual`
- `preferred_answer_evidence_hit_rate`
- `relation_schema_conformance_rate`
- `fallback_error_rate`

Primary reported runs should include:

- overall results
- category breakdown
- confidence intervals when available

## Current Bottlenecks To Address Next

Current experimental results suggest that graph structure improvements are not automatically transferring to final answer quality.

The next implementation work should address these bottlenecks in this priority order:

1. redefine `query_support_path_coverage` using question-intent-specific path requirements
2. make graph answers consume selected triples directly, not only graph-biased sentence reranking
3. replace coarse induced extraction gold with a small manual gold subset before attempting broader evaluator redesign

Do not try to solve all evaluation weaknesses at once. Fix the measurement and evidence-consumption path first.

## Query-Support Coverage Redefinition

Do **not** define coverage only as the presence of profile-level default relation types.

Coverage must move toward a question-intent-specific definition:

- `intent_id`
- `required_relations`
- `required_entity_types`
- `focus_slots`

Coverage should mean:

- whether the graph contains the subgraph required by the question intent
- whether the required slots or paths for that intent are populated

Start with a lightweight approach:

- use a small curated intent set
- prefer rule-based or lightly prompted intent mapping
- do not build a large intent-classification subsystem

## Graph Answering Requirement

Graph-based answering must not treat graph edges only as keyword hints for sentence reranking.

The preferred direction is:

- convert selected edges into a compact evidence bundle
- include canonical node names, relation types, provenance snippets, and confidence
- generate answers from that evidence bundle
- use reference text only as secondary support

Minimal acceptable behavior:

- if required slots are present, answer from the structured graph evidence
- if some required slots are missing, answer conservatively and state what is grounded vs missing

Do not build a large agentic answer synthesis system. Keep this step simple and inspectable.

## Extraction Evaluation Guidance

The current induced extraction gold may undercount ontology-friendly or more specific extractions.

Before implementing a broad canonicalized matching system, prefer a **small manual gold subset**:

- 10-15 examples per category
- 30-45 examples total
- gold profile
- gold triples
- required answer slots
- preferred answer evidence

This subset should be used to validate whether ontology-constrained extraction is genuinely worse, or whether the current evaluator proxy is too coarse.

Do not invest first in a large normalization engine unless the small manual gold subset shows it is necessary.

## Minimal Metadata Only

Preserve only metadata needed to support the four target measurements.

Allowed metadata:

- profile confidence
- extraction confidence
- link confidence
- schema validity
- query-support flags or path support annotations

Do not build a large metadata framework.

## Required Baselines

Keep baseline reporting split into two groups.

Auxiliary non-graph baselines:

1. `question_only_baseline`
2. `reference_only_baseline`

Primary ontology-guided baselines:

1. `graph_without_ontology_constraints`
2. `graph_with_profile_selection_only`
3. `graph_with_profile_plus_constrained_extraction_and_linking`
4. `full_minimal_pipeline_with_quality_aware_evidence_selection`

Do not build a full combinatorial grid over all stages.

The evaluation should remain hierarchical and interpretable.
Main comparison tables should emphasize the ontology-guided baselines, while the
two non-graph baselines remain auxiliary sanity checks.

## Evaluation Fairness And Sampling

The source dataset may be imbalanced across categories. Do **not** use raw category frequency as the evaluation design.

For tutorial experiments, evaluation must use a **category-balanced slice** across:

- Governance
- Financials
- Shareholder Return

Rules:

- sample the same number of documents or chunks per category for the main evaluation
- sample the same number of questions per category for the main evaluation
- run all main baselines on the same balanced slice
- report category-level results first
- report macro averages across categories as the primary aggregate
- report micro averages only as secondary context

If runtime is high, use staged evaluation:

1. a small pilot slice to verify the pipeline end-to-end
2. a balanced main evaluation slice for all reported tutorial results

Do not claim cross-category comparisons from an unbalanced sample.

The goal is controlled causal comparison, not full-distribution estimation.

## Shared Answer-Agent Rule

For the main experiment, answer generation must use a shared answer-agent
design.

Keep fixed across baselines:

- answer instruction
- answer model
- answer output schema
- answer context budget

Only change:

- question-only context
- reference-only context
- graph evidence bundle context

The legacy heuristic answer synthesis path may remain available only as a
supplemental answer-layer comparison.

Do not present heuristic answer synthesis as the main result unless explicitly
requested.

## Prompt And Evidence Discipline

Graph evidence should be serialized in a stable, inspectable format.

Preferred sections:

- `intent`
- `entities`
- `triples`
- `provenance snippets`
- `missing slots`

If these sections change, update:

- answer serialization code
- docs under `docs/reviews/`
- any proposal-facing figures or tables that explain the pipeline

## Reporting And Interpretation Rules

Main tutorial-facing artifacts should make these comparisons easy to inspect:

- naive graph vs ontology-guided graph
- profile selection only vs profile + constrained extraction
- constrained graph vs full pipeline

`reference_only_baseline` should usually remain auxiliary in interpretation:

- keep it in the experiment
- use it to interpret residual grounding gaps
- do not let it dominate the main graph-ontology narrative

## Artifact Expectations

Whenever a new main run is used for proposal-facing reporting, regenerate:

- proposal metrics report
- readable snapshot
- proposal figures (`Figure 1`, `Chart 1`, `Table 1`) if affected

Do not hand-edit numerical assets if they can be regenerated from scripts.

## Expected Artifacts

The implementation should produce outputs that let us inspect:

- profile decisions
- extracted entities and relations
- canonical nodes and graph edges
- graph quality issues
- query-support path status
- answers under each baseline setting
- comparison tables for the four target metrics
- category-wise summaries and macro-average summaries for the balanced evaluation slice

## Success Criteria

The implementation is successful if it makes the following visible:

- whether ontology/profile selection helped
- whether ontology-constrained extraction helped
- whether graph quality improved
- whether that improvement transferred to downstream answers

If a change does not make one of those points more visible, it is probably unnecessary.

## Failure Interpretation

Failures are useful if they remain interpretable.

The implementation should make it possible to explain failure in terms of:

- profile mismatch
- extraction errors
- linking errors
- schema violations
- disconnected graph structure
- missing query-support paths
- coarse query-intent coverage definitions
- evidence-selection bottlenecks
- answer-synthesis bottlenecks
- evaluator proxy limitations

Do not optimize only for higher final scores. Optimize for interpretability of the pipeline.

## Finance Scope

Keep the case study limited to:

- Governance
- Financials
- Shareholder Return

Do not expand the domain scope unless explicitly requested.

## Final Instruction

When in doubt:

1. remove complexity
2. preserve the causal chain
3. keep only what supports the four target metrics
