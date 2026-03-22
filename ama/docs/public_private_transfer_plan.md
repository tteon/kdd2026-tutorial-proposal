# Public / Private Transfer Plan

## Goal

Move the reusable code and reproducibility scaffold to GitHub, while keeping
research results, internal notes, and sensitive runtime outputs in a private
shared folder such as Google Drive.

## Put on GitHub

Public-safe code and reproducibility files:

- `debate/`
- `experiments/`
- `framework/`
- `tracing/`
- `docs/migration_handoff.md`
- `AGENTS.md`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `docker-compose.yml`
- `.env.example`
- `.gitignore`

Public-safe supporting docs only if reviewed:

- `CODEBASE_MAP.md`
- `docs/current_pipeline_runtime.md`
- `docs/ama_experiment_architecture.drawio`

## Keep private in Google Drive

Do not push these to a public repo:

- `artifacts/`
- `logs/`
- `.beads/`
- `data/neo4j/`
- `data/fibo_cache/`
- internal planning / execution memory docs
- experiment interpretation notes tied to unpublished results
- Opik trace exports / screenshots / trace IDs
- private context PDFs and local reference material

Examples that should remain private unless explicitly sanitized:

- `Documentation.md`
- `Plan.md`
- `Implement.md`
- `RunPrompt.md`
- `Prompt.md`
- `docs/finder_category_signal_analysis.md`
- `docs/finder_category_signal_analysis.json`
- `docs/finder_analysis_visuals.md`
- `docs/finder_rdf_lpg_presentation_brief.md`
- `docs/ontology_ablation_report.md`
- `docs/ama_master_report.md`
- `context-*.pdf`
- `context-*.md`

## Operational split

- GitHub = code, setup, contract, reproducibility frame
- Google Drive = experiment outputs, evaluations, private notes, screenshots

## Recommended branch content

For the first migration branch, prefer a minimal public-safe payload:

- code directories
- setup files
- handoff doc
- cleaned README

Do not include experiment outputs or detailed unpublished findings in the first
public branch.
