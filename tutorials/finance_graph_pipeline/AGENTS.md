# AGENTS.md

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

Only compare the following evaluation settings:

1. `text_only_baseline`
2. `graph_without_ontology_constraints`
3. `graph_with_profile_selection_only`
4. `graph_with_profile_plus_constrained_extraction_and_linking`
5. `full_minimal_pipeline_with_quality_aware_evidence_selection`

Do not build a full combinatorial grid over all stages.

The evaluation should be hierarchical and interpretable, not exhaustive.

## Expected Artifacts

The implementation should produce outputs that let us inspect:

- profile decisions
- extracted entities and relations
- canonical nodes and graph edges
- graph quality issues
- query-support path status
- answers under each baseline setting
- comparison tables for the four target metrics

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
- evidence-selection bottlenecks

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
