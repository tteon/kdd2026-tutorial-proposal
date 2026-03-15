# Pre-Submission Pipeline And Experiment Review

This note is the pre-submission review for the current finance graph tutorial
companion. It is written to answer one question before proposal submission:

Does the code currently implement the intended experimental story, and do the
current results support the claim at the right level?

## 1. Executive Summary

Short answer: mostly yes.

The current code now implements the intended minimal causal chain:

1. ontology/profile selection
2. ontology-constrained extraction
3. entity linking and graph construction
4. graph quality and query-support analysis
5. shared answer-agent generation
6. comparison against FinDER ground-truth answers

The most important design correction is already in place:

- the main experiment now uses one shared answer agent across all baselines
- the answer prompt, model, output style, and context budget are fixed
- only the provided context changes

This means the answerer is no longer the main confound.

The strongest current result is:

- ontology-guided graph baselines clearly outperform naive graph baselines
- but they still do not beat the `reference_only` text baseline on the current
  `50/category` primary run

That is still a valid tutorial result. It supports the claim that graph
construction quality matters, while also showing that graph quality does not
automatically exceed strong text-reference baselines.

## 2. Intended Problem, Claim, And Scope

### 2.1 Problem Statement

The tutorial is not trying to build the largest finance agent system.

It is trying to show a narrower point:

`ontology choice -> extraction quality -> graph quality -> downstream answer quality`

In other words, we want to show that domain-specific LLM behavior depends on
whether the system can build a query-supporting graph with enough ontology
alignment and graph quality.

### 2.2 Dataset Scope

We intentionally restrict the case study to the FinDER top-3 categories:

- Governance
- Financials
- Shareholder Return

This is aligned with the original design intent: these are the three areas
where FIBO-style semantic constraints are most likely to make a visible
difference.

### 2.3 Practical Claim For The Proposal

The safe version of the current claim is:

Ontology-guided graph construction materially improves graph structure and
downstream answer quality relative to naive graph construction, but strong
text-reference baselines remain competitive and expose the remaining grounding
gaps in the graph pipeline.

That is narrower than "graph beats text," but it is well supported by the
current code and results.

## 3. What Is Implemented In Code

### 3.1 Storage And Runtime Stack

The current stack is:

- DozerDB for graph storage
- SQLite for run metadata and answer records
- JSON and CSV exports for reports and review artifacts
- OpenAI Agents SDK for profile selection, extraction, and answering

Main infra files:

- `docker-compose.yml`
- `docker/python/Dockerfile`
- `scripts/init_metadata_db.py`
- `scripts/filter_finder_top3.py`

### 3.2 Main Pipeline Components

The runnable implementation is centered in:

- `tutorials/finance_graph_pipeline/framework/finder_experiment.py`
- `tutorials/finance_graph_pipeline/framework/agents.py`
- `tutorials/finance_graph_pipeline/framework/quality.py`
- `tutorials/finance_graph_pipeline/framework/intents.py`
- `tutorials/finance_graph_pipeline/framework/agent_worker.py`

Current stage mapping:

1. Data loading
   - FinDER subset loading and per-category sampling
   - source text comes from FinDER `references`

2. Profile selection
   - one selected profile from `governance`, `financials`,
     `shareholder_return`
   - OpenAI mode or heuristic mode

3. Ontology-constrained extraction
   - allowed entity types and relation types are profile-specific
   - extractor returns typed entities and typed relations

4. Entity linking and graph materialization
   - extracted entities are normalized into graph nodes
   - relations are written into the graph and logged in SQLite

5. Query-support analysis
   - current code uses question-intent-specific requirements rather than only
     coarse profile-level requirements
   - this is the right design direction for the tutorial

6. Evidence selection
   - graph edges are selected and turned into a compact evidence bundle

7. Answer generation
   - one fixed answer agent is used across baselines
   - this is now the main answer path

### 3.3 Shared Answer-Agent Design

This part now matches the intended experiment.

The answerer is fixed across:

- `question_only_baseline`
- `reference_only_baseline`
- all graph baselines

What changes is only the context.

Current context policy:

- `question_only`
  - question only
  - explicit note that no supporting evidence is available

- `reference_only`
  - question
  - selected reference text only
  - same answer instruction and shared context budget

- graph baselines
  - question
  - intent id
  - entities
  - triples
  - provenance snippets
  - missing slots

This design is fair and close to the intended causal comparison.

### 3.4 What Is Explicitly Not In Scope

The code intentionally does not make the following central:

- large intent classification subsystems
- multi-agent answer synthesis
- broad ontology extension
- UI or deployment workflow
- question router as the main story

This is correct for the proposal narrative.

## 4. Experiment Design

### 4.1 Baselines

The current comparison set is:

- `question_only_baseline`
- `reference_only_baseline`
- `graph_without_ontology_constraints`
- `graph_with_profile_selection_only`
- `graph_with_profile_plus_constrained_extraction_and_linking`
- `full_minimal_pipeline_with_quality_aware_evidence_selection`

Supplemental answer-layer comparison:

- `--answer-mode heuristic_synthesis`

The main claim should be made using the shared answer-agent setting, not the
legacy heuristic answer mode.

### 4.2 Main Metrics

The four primary metrics are:

- `profile_selection_accuracy`
- `ontology_constrained_extraction_f1`
- `query_support_path_coverage`
- `answer_quality_delta`

Here `answer_quality_delta` is interpreted against the baseline matrix, with
`reference_only` as the anchor for direct delta reporting.

### 4.3 Supporting Metrics

The additional supporting metrics are:

- `required_answer_slot_coverage_manual`
- `preferred_answer_evidence_hit_rate`
- `relation_schema_conformance_rate`
- `fallback_error_rate`

This is a good set for the proposal because it shows:

- graph structure quality
- evidence quality
- runtime robustness
- whether graph support is actually filling question-relevant slots

### 4.4 Evaluation Sizes

Current experiment sizes are:

- primary result: `50/category`
- robustness target: `90/category`
- manual gold subset: `30 examples` total now available for evaluation

This is enough for the proposal stage, with `50/category` as the main table and
`90/category` as robustness evidence.

## 5. Hypotheses

The current code and evaluation logic are testing these hypotheses.

### H1. Profile selection helps only if it changes downstream graph construction.

Expected result:

- `graph_with_profile_selection_only` should not improve much unless
  extraction/linking can use that profile correctly

### H2. Ontology constraints improve graph quality.

Expected result:

- higher schema conformance
- higher question-intent-specific coverage
- lower noise than unconstrained graph extraction

### H3. Better graph quality should improve answer quality relative to naive graph baselines.

Expected result:

- constrained graph baselines should beat unconstrained graph baselines

### H4. Quality-aware evidence selection only matters if it actually changes the evidence bundle.

Expected result:

- `full` should beat `profile+constrained+linking` only if quality metadata
  removes or downweights harmful evidence

## 6. Main Run Used For Review

Primary shared answer-agent run:

- Run ID: `finder-20260315013504-2f8ea4d3`
- Summary:
  `data/experiment_runs/finder-20260315013504-2f8ea4d3_summary.json`
- Readable snapshot:
  `exports/readable_results/2026-03-15_intermediate-shared-answer-agent_50-per-category_completed_snapshot__run-finder-20260315013504-2f8ea4d3.md`
- Snapshot figure:
  `exports/readable_results/2026-03-15_intermediate-shared-answer-agent_50-per-category_completed_snapshot__run-finder-20260315013504-2f8ea4d3.svg`
- Proposal report:
  `exports/proposal_metrics/finder-20260315013504-2f8ea4d3_proposal_metrics_report.json`
- Main metrics CSV:
  `exports/proposal_metrics/finder-20260315013504-2f8ea4d3_proposal_main_metrics.csv`
- Manual gold evaluation:
  `exports/manual_gold/finder-20260315013504-2f8ea4d3_manual_gold_eval.json`

## 7. Main Results

### 7.1 Overall Results

| Baseline | Answer Quality | Delta vs Reference | Profile Acc | Extraction F1 | Coverage |
| --- | --- | --- | --- | --- | --- |
| `question_only_baseline` | 0.159 | -0.201 |  |  |  |
| `reference_only_baseline` | 0.360 | 0.000 |  |  |  |
| `graph_without_ontology_constraints` | 0.175 | -0.185 |  | 0.126 | 0.000 |
| `graph_with_profile_selection_only` | 0.175 | -0.185 | 0.900 | 0.104 | 0.000 |
| `graph_with_profile_plus_constrained_extraction_and_linking` | 0.318 | -0.042 | 0.900 | 0.083 | 0.589 |
| `full_minimal_pipeline_with_quality_aware_evidence_selection` | 0.321 | -0.039 | 0.900 | 0.083 | 0.589 |

### 7.2 Confidence Intervals For The Main Comparisons

`reference_only_baseline`

- `answer_quality_score = 0.360 [0.341, 0.379]`

`graph_without_ontology_constraints`

- `answer_quality_score = 0.175 [0.165, 0.184]`
- `answer_quality_delta = -0.185 [-0.203, -0.168]`
- `ontology_constrained_extraction_f1 = 0.126 [0.103, 0.151]`

`graph_with_profile_selection_only`

- `profile_selection_accuracy = 0.900 [0.847, 0.940]`
- `answer_quality_score = 0.175 [0.166, 0.184]`
- `answer_quality_delta = -0.185 [-0.202, -0.167]`

`graph_with_profile_plus_constrained_extraction_and_linking`

- `answer_quality_score = 0.318 [0.297, 0.338]`
- `answer_quality_delta = -0.042 [-0.064, -0.022]`
- `query_support_path_coverage = 0.589 [0.537, 0.638]`
- `relation_schema_conformance_rate = 1.000`

`full_minimal_pipeline_with_quality_aware_evidence_selection`

- `answer_quality_score = 0.321 [0.300, 0.341]`
- `answer_quality_delta = -0.039 [-0.060, -0.018]`
- `query_support_path_coverage = 0.589 [0.537, 0.638]`
- `fallback_error_rate = 0.007 [0.000, 0.020]`

### 7.3 Category Breakdown

`Financials`

- `reference_only = 0.394`
- `graph_without_ontology_constraints = 0.183`
- `graph_with_profile_selection_only = 0.181`
- `graph_with_profile_plus_constrained_extraction_and_linking = 0.389`
- `full = 0.386`

Interpretation:

- This is the strongest category.
- Ontology-guided graph answers almost match the reference-only baseline.
- Coverage is already high here and the graph is structurally useful.

`Governance`

- `reference_only = 0.294`
- `graph_without_ontology_constraints = 0.163`
- `graph_with_profile_selection_only = 0.165`
- `graph_with_profile_plus_constrained_extraction_and_linking = 0.234`
- `full = 0.239`

Interpretation:

- Graph baselines improve a lot over naive graph extraction.
- But governance still trails the text-reference baseline by a visible margin.
- The remaining issue is likely question-specific role/committee grounding.

`Shareholder Return`

- `reference_only = 0.391`
- `graph_without_ontology_constraints = 0.178`
- `graph_with_profile_selection_only = 0.180`
- `graph_with_profile_plus_constrained_extraction_and_linking = 0.330`
- `full = 0.339`

Interpretation:

- This category benefits from ontology constraints.
- But it still underperforms `reference_only`, especially on authorization,
  remaining-capacity, and policy-vs-action distinctions.

## 8. Manual-Gold Check

The current main run was also checked on the local manual-gold subset:

- File:
  `exports/manual_gold/finder-20260315013504-2f8ea4d3_manual_gold_eval.json`

For both:

- `graph_with_profile_plus_constrained_extraction_and_linking`
- `full_minimal_pipeline_with_quality_aware_evidence_selection`

the manual-gold summary is nearly identical:

- `sample_count = 30`
- `profile_selection_accuracy_manual = 1.000`
- `intent_accuracy_manual = 0.733`
- `ontology_constrained_extraction_f1_manual = 0.219`
- `required_answer_slot_coverage_manual = 0.558`
- `preferred_answer_evidence_hit_rate = 0.928`
- `answer_token_f1 ~= 0.347`

Interpretation:

- The graph pipeline is much stronger under manual-gold judgment than the coarse
  induced extraction proxy suggests.
- Preferred evidence hit rate is very high.
- Slot coverage is only moderate.
- This supports the idea that the main remaining bottleneck is not simple
  evidence retrieval failure, but incomplete question-slot filling.

## 9. Code-Level Interpretation

### 9.1 What Is Clearly Working

1. Shared answer-agent fairness
   - This is now implemented correctly.
   - The answerer is fixed and only context changes.

2. Ontology-guided graph construction
   - The pipeline clearly separates unconstrained graph extraction from
     profile-conditioned constrained extraction.

3. Graph-quality evaluation
   - Coverage, schema conformance, and fallback/error tracking are implemented.

4. Manual-gold evaluation
   - The infrastructure exists and is useful for interpretation.

### 9.2 What The Results Say About The Pipeline

1. H1 is supported.
   - `graph_with_profile_selection_only` is almost identical to the
     unconstrained graph baseline on answer quality.
   - This means profile routing by itself is not enough.

2. H2 is supported.
   - Constrained graph baselines achieve `coverage = 0.589` and
     `schema_conformance = 1.0`, while unconstrained graph baselines have
     `coverage = 0.0` and `schema_conformance = 0.0`.

3. H3 is supported in a limited but important sense.
   - Ontology-guided graph baselines substantially beat naive graph baselines.
   - They do not yet beat `reference_only`.

4. H4 is only weakly supported.
   - `full` only improves from `0.318` to `0.321`.
   - Quality-aware evidence selection is present, but it is not yet a strong
     differentiator in this run.

### 9.3 Why Extraction F1 Looks Lower While Answers Improve

This is an important point for proposal writing.

The current result shows:

- unconstrained extraction F1 is higher than constrained extraction F1
- but constrained graph answering is much better

This does not mean ontology constraints are harmful.

It means the current induced extraction F1 proxy is coarse and string-sensitive,
while downstream answer utility benefits from semantically cleaner graph
structure. The manual-gold extraction F1 of `0.219` is a better signal than the
coarse overall proxy of `0.083`.

## 10. Does The Implementation Match The Intended Design?

### 10.1 Yes

These parts now match the intended design well:

- shared answer agent across baselines
- graph evidence serialized with structure preserved
- answer evaluation against FinDER ground-truth answers
- top-3 FinDER focus
- ontology-guided vs naive graph comparison
- graph quality plus downstream quality joint evaluation

### 10.2 Partially

These parts are present but still weak:

- quality-aware evidence selection
  - implemented, but currently small marginal gain

- manual-gold integration into the standard reporting loop
  - implemented now, but still needs to be treated as a first-class review
    artifact in the submission workflow

- question-intent slot filling
  - implemented in the evaluation logic, but the actual graph answer behavior
    still leaves many required slots empty

### 10.3 Not Yet Fully Proven

These are not yet complete enough to overclaim:

- "graph beats strong text-reference baseline"
- "quality-aware evidence selection is a major gain"
- "current extraction F1 proxy is a definitive extraction metric"

## 11. Safe Proposal Narrative

The proposal should probably say something close to the following:

1. The tutorial shows that ontology-guided graph construction materially changes
   graph structure and query-support behavior.
2. In the finance case study, ontology-guided graph pipelines substantially
   improve answer quality relative to naive graph construction.
3. Strong text-reference baselines remain competitive, which makes the tutorial
   more realistic and highlights the remaining gap between graph structure and
   answer grounding.
4. The tutorial therefore teaches not only how to build ontology-grounded
   graphs, but also how to evaluate where graph quality does and does not
   translate into downstream generation gains.

This is strong, accurate, and consistent with the code.

## 12. Remaining Risks Before Final Submission

1. The `90/category` robustness result should still be checked before treating
   the main trend as fully stable.
2. The legacy answer-mode comparison can stay supplemental, but it should not
   distract from the shared answer-agent main story.
3. Some proposal readers may over-focus on the low coarse extraction F1.
   The proposal should therefore pair it with manual-gold and schema/coverage
   evidence.
4. The proposal should avoid promising that the current pipeline already solves
   answer synthesis. The evidence shows progress, not final superiority over
   strong text baselines.

## 13. Reviewer Checklist

If this document is used for final review, the main questions to answer are:

1. Is the tutorial claim written narrowly enough?
2. Is the shared answer-agent setup the only main answer comparison shown?
3. Are the proposal tables based on `50/category` with CI and category
   breakdown?
4. Is the manual-gold subset explicitly used to explain the extraction-F1
   mismatch?
5. Is the `reference_only` baseline framed as a strong comparison point rather
   than an embarrassing failure case for graphs?

If the answer is yes to those five questions, the current implementation is in
good enough shape for proposal submission.
