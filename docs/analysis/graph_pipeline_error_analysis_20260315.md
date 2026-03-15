# Graph Pipeline Error Analysis (2026-03-15)

Target run:
- `finder-20260314152400-53f4e7e5`
- Summary: `data/experiment_runs/finder-20260314152400-53f4e7e5_summary.json`
- Export: `exports/finder_runs/finder-20260314152400-53f4e7e5_export.json`

## Key findings

1. `query_support_path_coverage` is overstated for many examples.
- Coverage is computed from profile-level fixed relation requirements, not question-intent-specific requirements.
- Code path:
  - `FinDERExample.query_template` maps every example in a category to one fixed template.
  - `GraphQualityAnalyzer.template_requirements` maps each fixed template to a static relation set.
- Result:
  - 73/90 examples had `coverage >= 0.5`.
  - Among those 73 examples, 27 were worse than `reference_only`, 40 tied, and only 6 improved.
- Interpretation:
  - The graph often contains some profile-consistent relations, but not necessarily the subgraph needed for the actual question.

2. Graph evidence is not consumed as structured evidence by the answerer.
- Current `FinDER` answer generation does not synthesize an answer from triples.
- It reranks reference sentences using question-token overlap plus node-name overlap from selected edges, then concatenates the top 2 sentences.
- Code path:
  - `EvidenceRetriever.select_reference_sentences`
  - `_graph_answer`
- Consequence:
  - The graph acts as a sentence-selection hint, not as a structured evidence substrate.
  - If the selected edges pull the wrong local section of the reference text, the answer degrades even when coverage is non-zero.

3. `full_minimal_pipeline_with_quality_aware_evidence_selection` is effectively identical to `graph_with_profile_plus_constrained_extraction_and_linking`.
- Artifact comparison across all 90 examples showed:
  - same answers: 90/90
  - same selected edges: 90/90
  - same query support records: 90/90
- Root cause:
  - `quality_aware=True` only drops edges with edge-level `quality_issues`.
  - In this run, the only quality issue type was `disconnected_node` on nodes.
  - Edge-level issue count was 0, so the quality-aware path filtered nothing.

4. `ontology_constrained_extraction_f1` is depressed by evaluation design as much as by extraction quality.
- The current metric uses an induced proxy gold extracted from `question_text + answer_text`.
- That proxy is coarse and string-sensitive.
- Typical mismatch modes:
  - generic gold vs specific predicted entity
  - coarse period gold vs explicit date predicted period
  - generic `company` subject in gold vs named program/entity subject in prediction
- Result:
  - constrained extraction often looks worse because it is more specific and more verbose than the proxy gold.

## Supporting numbers

- `reference_only_baseline.answer_quality_score = 0.225`
- `full_minimal_pipeline.answer_quality_score = 0.205`
- `full_minimal_pipeline.query_support_path_coverage = 0.617`
- `graph_with_profile_plus_constrained_extraction_and_linking.answer_quality_score = 0.205`
- `graph_with_profile_plus_constrained_extraction_and_linking.ontology_constrained_extraction_f1 = 0.094`

By category:
- Financials:
  - `avg_support = 0.983`
  - `avg_delta(full - reference) = -0.007`
- Governance:
  - `avg_support = 0.400`
  - `avg_delta(full - reference) = -0.008`
- Shareholder return:
  - `avg_support = 0.467`
  - `avg_delta(full - reference) = -0.043`

## Representative examples

### Governance: `aaa2a2aa`
- Question asks about Catherine R. Clay's role transition.
- Graph answer follows a broad executive roster and loses the role-transition focus.
- `full token_f1 = 0.101`
- `reference_only token_f1 = 0.283`
- Interpretation:
  - The graph captures governance relations, but not the question-specific path needed for the answer.

### Financials: `8ac98b17`
- Question asks about CRWD liquidity breakdown as of January 31, 2024.
- Graph answer includes deferred revenue and other nearby financial facts.
- `full token_f1 = 0.301`
- `reference_only token_f1 = 0.394`
- Interpretation:
  - Graph-selected evidence expanded the local evidence region instead of narrowing it to the liquidity-specific slots.

### Shareholder return: `b1e82d10`
- Question asks for remaining repurchase balance and future capital return drivers.
- Graph answer focuses on repurchase execution amounts and program history.
- `full token_f1 = 0.258`
- `reference_only token_f1 = 0.360`
- Interpretation:
  - The selected triples are relevant to shareholder return broadly, but not specific enough to the requested remaining-balance slot.

### Shareholder return: `864fd74a`
- Question asks about the IRA excise tax impact.
- Graph answer pivots to repurchase authorization and dividends rather than the tax effect.
- `full token_f1 = 0.166`
- `reference_only token_f1 = 0.245`

## Why constrained extraction F1 drops

Aggregate extraction behavior:
- `graph_without_ontology_constraints`
  - average entities: `9.92`
  - average relations: `10.03`
  - average entity F1: `0.099`
  - average relation F1: `0.189`
- `graph_with_profile_plus_constrained_extraction_and_linking`
  - average entities: `18.09`
  - average relations: `19.2`
  - average entity F1: `0.076`
  - average relation F1: `0.111`

Important evaluator artifact:
- Gold relation count histogram on the 90-example run:
  - `0 relations`: 36 examples
  - `1 relation`: 46 examples
  - `2 relations`: 5 examples
- Governance alone contributes 27 examples with `0 gold relations`.

Implication:
- The induced gold is sparse.
- Constrained extraction is detailed.
- Detailed predictions are punished heavily by exact/set-based matching against a sparse proxy gold.

## Current answer input format

The `FinDER` run path currently gives the answerer:
- `selected_edge_ids`
- node-name-derived lexical hints
- the original reference text

It does **not** currently pass:
- canonicalized triples as a structured bundle
- slot completeness
- grounded vs missing fields
- provenance per triple

That is why graph evidence has limited leverage over the final answer quality.

## Concrete changes implied by this analysis

1. Replace profile-level coverage with question-intent-specific path requirements.
2. Replace sentence reranking answer generation with evidence-bundle-driven synthesis.
3. Replace induced proxy extraction F1 with a small manual gold subset before building any large canonicalization layer.
