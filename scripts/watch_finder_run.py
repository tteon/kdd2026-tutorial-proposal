from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a FinDER run checkpoint and append status snapshots.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--poll-seconds", "--interval-seconds", dest="poll_seconds", type=int, default=30)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--output-log", default=None)
    parser.add_argument("--run-export-on-complete", "--export-on-complete", dest="run_export_on_complete", action="store_true")
    parser.add_argument("--run-manual-eval-on-complete", action="store_true")
    parser.add_argument("--run-proposal-report-on-complete", action="store_true")
    parser.add_argument(
        "--manual-eval-run-id",
        default=None,
        help="Run id whose manual-gold eval should be attached to the proposal report. Defaults to --run-id.",
    )
    return parser.parse_args()


def _summarize(payload: dict) -> dict:
    prefetch = payload.get("prefetch_progress", {})
    baseline_progress = payload.get("baseline_progress", {})
    comparison = payload.get("available_comparison_table") or payload.get("comparison_table") or []
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "stage": payload.get("stage"),
        "current_baseline": payload.get("current_baseline"),
        "sample_count": payload.get("sample_count"),
        "prefetch_progress": prefetch,
        "baseline_progress": {
            baseline: {
                "processed_examples": info.get("progress", {}).get("processed_examples"),
                "total_examples": info.get("progress", {}).get("total_examples"),
                "answer_quality_score": info.get("metrics", {}).get("answer_quality_score"),
                "ontology_constrained_extraction_f1": info.get("metrics", {}).get("ontology_constrained_extraction_f1"),
                "query_support_path_coverage": info.get("metrics", {}).get("query_support_path_coverage"),
            }
            for baseline, info in baseline_progress.items()
        },
        "available_comparison_table": comparison,
    }


def _emit_snapshot(output_log: Path, snapshot: dict) -> None:
    with output_log.open("a") as handle:
        handle.write(json.dumps(snapshot) + "\n")
    print(json.dumps(snapshot, indent=2), flush=True)


def _run_completion_step(
    output_log: Path,
    step_name: str,
    command: list[str],
) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    snapshot = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completion_step",
        "step": step_name,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
    _emit_snapshot(output_log, snapshot)


def main() -> int:
    args = parse_args()
    checkpoint_path = ROOT_DIR / "data" / "experiment_runs" / f"{args.run_id}_checkpoint.json"
    summary_path = ROOT_DIR / "data" / "experiment_runs" / f"{args.run_id}_summary.json"
    output_log = Path(args.output_log) if args.output_log else ROOT_DIR / "exports" / "finder_runs" / f"{args.run_id}_monitor.jsonl"
    output_log.parent.mkdir(parents=True, exist_ok=True)

    last_mtime: float | None = None
    last_payload_status: str | None = None
    exported = False
    stale_alert_emitted_for_mtime: float | None = None

    while True:
        source = checkpoint_path if checkpoint_path.exists() else summary_path
        if source.exists():
            mtime = source.stat().st_mtime
            if last_mtime is None or mtime > last_mtime:
                last_mtime = mtime
                stale_alert_emitted_for_mtime = None
                payload = json.loads(source.read_text())
                snapshot = _summarize(payload)
                _emit_snapshot(output_log, snapshot)

                status = payload.get("status")
                last_payload_status = status
                if status in {"completed", "failed"} or source == summary_path:
                    if args.run_export_on_complete and not exported:
                        _run_completion_step(
                            output_log,
                            "export_finder_run_results",
                            [
                                sys.executable,
                                str(ROOT_DIR / "scripts" / "export_finder_run_results.py"),
                                "--run-id",
                                args.run_id,
                            ],
                        )
                        exported = True
                    if status == "completed" and args.run_manual_eval_on_complete:
                        _run_completion_step(
                            output_log,
                            "evaluate_manual_gold_subset",
                            [
                                sys.executable,
                                str(ROOT_DIR / "scripts" / "evaluate_manual_gold_subset.py"),
                                "--run-id",
                                args.run_id,
                                "--output",
                                str(ROOT_DIR / "exports" / "manual_gold" / f"{args.run_id}_manual_gold_eval.json"),
                            ],
                        )
                    if status == "completed" and args.run_proposal_report_on_complete:
                        manual_eval_run_id = args.manual_eval_run_id or args.run_id
                        _run_completion_step(
                            output_log,
                            "build_proposal_metrics_report",
                            [
                                sys.executable,
                                str(ROOT_DIR / "scripts" / "build_proposal_metrics_report.py"),
                                "--run-id",
                                args.run_id,
                                "--manual-eval-run-id",
                                manual_eval_run_id,
                                "--output-dir",
                                "exports/proposal_metrics",
                            ],
                        )
                    return 0
            elif (
                last_mtime is not None
                and last_payload_status not in {"completed", "failed"}
                and args.stale_seconds > 0
                and time.time() - last_mtime >= args.stale_seconds
                and stale_alert_emitted_for_mtime != last_mtime
            ):
                stale_alert_emitted_for_mtime = last_mtime
                snapshot = {
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "run_id": args.run_id,
                    "status": "stalled",
                    "stage": "watchdog",
                    "current_baseline": None,
                    "sample_count": None,
                    "stale_seconds": round(time.time() - last_mtime, 1),
                    "last_checkpoint_path": str(source),
                    "last_checkpoint_mtime_utc": datetime.fromtimestamp(last_mtime, tz=timezone.utc).isoformat(),
                    "message": "Checkpoint has not updated within the configured stale threshold.",
                }
                _emit_snapshot(output_log, snapshot)

        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
