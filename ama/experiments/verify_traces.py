"""
Verify trace completeness from an experiment manifest.

Usage:
    uv run python -m experiments.verify_traces \
        --manifest artifacts/rdf_vs_lpg/manifest.json

Reads the manifest and each artifact JSON, checking:
  - trace_id present
  - status field present
  - error_notes
  - whether the artifact has execution-complete fields

This script intentionally separates:
  - execution completeness: did the run produce a trace/artifact?
  - task success signals: did it extract entities or materialize a graph?

Empty outputs can still represent valid, inspectable model failures.

Outputs a summary report to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify trace completeness from manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    total = len(manifest.get("examples", []))
    execution_complete = 0
    errors = 0
    missing_trace = 0
    missing_status = 0
    execution_incomplete = 0
    empty_entities = 0
    empty_graph = 0

    print(f"Manifest: {manifest_path}")
    print(f"Total examples: {total}\n")

    for entry in manifest.get("examples", []):
        artifact_path = Path(entry["artifact_path"])
        if not artifact_path.exists():
            print(f"  MISSING: {artifact_path}")
            errors += 1
            continue

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        status = artifact.get("status")
        trace_id = artifact.get("trace_id", "")
        entities = artifact.get("extracted_entities", [])
        linked = artifact.get("linked_entities", [])
        graph = artifact.get("graph_preview", {})
        error_notes = artifact.get("error_notes", [])

        has_graph = bool(graph.get("triples") or graph.get("nodes") or graph.get("edges"))

        issues = []
        if not trace_id:
            issues.append("no_trace_id")
            missing_trace += 1
        if not status:
            issues.append("no_status")
            missing_status += 1
        if not entities:
            issues.append("empty_entities")
            empty_entities += 1
        if not has_graph:
            issues.append("empty_graph")
            empty_graph += 1
        if error_notes:
            issues.append(f"errors={len(error_notes)}")

        execution_ok = bool(trace_id and status and not error_notes)
        if execution_ok:
            execution_complete += 1
        else:
            errors += 1
            execution_incomplete += 1

        label = "EXEC_OK" if execution_ok else "EXEC_INCOMPLETE"
        if issues:
            label = f"{label} | " + ", ".join(issues)
        rep = entry.get("representation", "?")
        eid = entry.get("example_id", "?")
        print(f"  [{rep}] {eid}: {label} (entities={len(entities)}, linked={len(linked)})")

    print(f"\n--- Summary ---")
    print(f"Execution complete: {execution_complete}/{total}")
    print(f"Execution incomplete: {execution_incomplete}/{total}")
    print(f"Artifacts with errors: {errors}/{total}")
    print(f"Missing trace IDs: {missing_trace}")
    print(f"Missing status:    {missing_status}")
    print(f"Empty entities:    {empty_entities}")
    print(f"Empty graph:       {empty_graph}")

    rate = execution_complete / total if total > 0 else 0.0
    print(f"Execution completeness rate: {rate:.1%}")


if __name__ == "__main__":
    main()
