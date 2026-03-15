# Scripts Index

Active experiment scripts:

- `filter_finder_top3.py`
  Filters FinDER down to the three target categories.
- `init_metadata_db.py`
  Initializes the SQLite metadata store.
- `run_finder_experiment.py`
  Main FinDER experiment runner.
- `evaluate_manual_gold_subset.py`
  Evaluates a run against the manual-gold subset.
- `analyze_finder_graph_baselines.py`
  Computes graph-structure and schema-conformance analysis.
- `build_proposal_metrics_report.py`
  Produces proposal-ready metrics with mean, 95% CI, and category breakdown.
- `export_finder_run_results.py`
  Exports a run into JSON/CSV snapshots.
- `print_finder_run_status.py`
  Prints checkpoint or summary status for a run.
- `watch_finder_run.py`
  Watches a long-running run and can trigger export/report generation on completion.

Recommended operator flow:

1. `filter_finder_top3.py`
2. `init_metadata_db.py`
3. `run_finder_experiment.py`
4. `evaluate_manual_gold_subset.py`
5. `build_proposal_metrics_report.py`
