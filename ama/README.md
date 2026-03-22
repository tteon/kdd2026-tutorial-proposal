# AMA Experiment Scaffold

This repository is set up for an observability-first comparison between RDF-oriented and LPG-oriented extraction/linking pipelines.

## Current scope

- FinDER on Hugging Face as the source dataset
- FIBO as the canonical linking target
- Opik traces with local JSON fallback when Cloud is unavailable
- shared artifact schema across RDF and LPG
- smoke-run runner that creates paired RDF/LPG artifacts on the same slice
- deterministic RDF/LPG materialization previews from the same linked-entity substrate
- semantic-hint translation layer that turns selected FIBO concepts into inspectable prompt rules

## Orientation

If you want a human-readable walkthrough of the code, start with `CODEBASE_MAP.md`.
If you want a diagram you can open and edit, see `docs/ama_experiment_architecture.drawio`.
If you want the current code-aligned runtime map, see `docs/current_pipeline_runtime.md`.

If you want to run Codex in a long-horizon, durable-memory style, use:

- `Prompt.md` — frozen target
- `Plan.md` — milestones and validations
- `Implement.md` — execution runbook
- `Documentation.md` — live status log
- `RunPrompt.md` — paste-ready kickoff prompt for a long-running Codex session

These files are execution memory, not a replacement for `AGENTS.md` or `EXPERIMENT_SETUP.md`.

## Why category duplicate counts can help

Counting repeated entities per category is a useful exploratory signal, but not a standalone measure of whether FIBO is "efficient".

It can help answer:

- whether a category naturally reuses a stable set of canonical concepts
- whether ontology-pack selection may be too broad or too narrow
- whether the linking layer is collapsing distinct mentions into the same concept too aggressively

It should be interpreted together with:

- coverage of linked entities
- null-link rate
- ambiguous-link rate
- canonicalization consistency

## Smoke runner

```bash
python -m experiments.rdf_vs_lpg_smoke --dataset-id <huggingface_dataset_id> --split train --sample-size 10
```

Ontology ablation is controlled with:

```bash
python -m experiments.rdf_vs_lpg_smoke --ontology-mode static
python -m experiments.rdf_vs_lpg_smoke --ontology-mode none
python -m experiments.rdf_vs_lpg_smoke --ontology-mode dynamic
```

## Prompt-rule inspection

```bash
python -m experiments.render_prompt_rules --category "Financials"
python -m experiments.render_prompt_rules --category "Risk" --ontology-mode dynamic --question "Who oversees cybersecurity risk?" --reference-text "The board and audit committee oversee cybersecurity risk management."
```

## Documentation entry points

- Start here for the consolidated project story: `docs/ama_master_report.md`
- Detailed FinDER lexical and phrase analysis: `docs/finder_category_signal_analysis.md`
- Visual tables and figures: `docs/finder_analysis_visuals.md`
- Presentation-style summary: `docs/finder_rdf_lpg_presentation_brief.md`
- Debate agent and router-policy spec: `docs/debate_agent_instruction_spec.md`

## Current experimental framing

- Treat ontology selection as part of the indexing pipeline, not external preprocessing
- Compare `no ontology` vs `static full ontology` vs `dynamic ontology slice`
- Use ontology-derived guidance packs rather than raw ontology dumps
- Manage prompt generation as an experimental condition:
  - ontology selection prompt
  - extraction prompt
  - linking prompt
  - relation extraction prompt
- Separate evaluation into:
  - ontology selection quality
  - ontology utility
  - extraction/linking quality

Required environment variables:

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`
- `FINDER_DATASET_ID` if not passed on the command line

Optional environment variable:

- `OPIK_PROJECT_NAME`

## Notes

- The current runner intentionally preserves the full trace/artifact contract before implementing extraction and linking.
- `select_ontology_context` is explicit and logged so later failures can be separated from extraction failures.
- Prompt rules are generated as semantic hints from selected FIBO concepts; they are not direct OWL axioms.
- If FinDER categories differ from the initial AGENTS categories, update the static FIBO pack mapping after the first smoke run.
