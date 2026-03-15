from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact status view for a FinDER run.")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = ROOT_DIR / "data" / "experiment_runs" / f"{args.run_id}_checkpoint.json"
    summary_path = ROOT_DIR / "data" / "experiment_runs" / f"{args.run_id}_summary.json"

    source_path = checkpoint_path if checkpoint_path.exists() else summary_path
    if not source_path.exists():
        raise SystemExit(f"No checkpoint or summary found for run {args.run_id}")

    payload = json.loads(source_path.read_text())
    print(f"run_id: {payload.get('run_id', args.run_id)}")
    print(f"source: {source_path}")
    print(f"status: {payload.get('status', 'completed' if source_path == summary_path else 'unknown')}")
    print(f"stage: {payload.get('stage', 'completed')}")
    print(f"current_baseline: {payload.get('current_baseline')}")
    print(f"sample_count: {payload.get('sample_count')}")
    prefetch = payload.get("prefetch_progress", {})
    if prefetch:
        print("prefetch_progress:")
        for key, value in prefetch.items():
            print(
                f"  {key}: {value.get('completed_examples')}/{value.get('total_examples')} "
                f"errors={value.get('error_count')}"
            )

    baseline_progress = payload.get("baseline_progress", {})
    if baseline_progress:
        print("baseline_progress:")
        for baseline in sorted(baseline_progress):
            progress = baseline_progress[baseline].get("progress", {})
            metrics = baseline_progress[baseline].get("metrics", {})
            print(
                "  "
                f"{baseline}: {progress.get('status')} "
                f"{progress.get('processed_examples')}/{progress.get('total_examples')} "
                f"answer={metrics.get('answer_quality_score')} "
                f"extract={metrics.get('ontology_constrained_extraction_f1')} "
                f"coverage={metrics.get('query_support_path_coverage')}"
            )

    table = payload.get("available_comparison_table") or payload.get("comparison_table") or []
    if table:
        print("comparison_table:")
        for row in table:
            print(
                "  "
                f"{row['baseline']}: "
                f"profile={row.get('profile_selection_accuracy')} "
                f"extract={row.get('ontology_constrained_extraction_f1')} "
                f"coverage={row.get('query_support_path_coverage')} "
                f"answer={row.get('answer_quality_score')} "
                f"delta={row.get('answer_quality_delta')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
