"""
Re-materialize all existing artifacts into DozerDB.

Usage:
    uv run python -m experiments.rematerialize_all [--clear-first]

Walks artifacts/<experiment>/{rdf,lpg}/*.json and loads each into the
appropriate DozerDB logical database (finderrdf or finderlpg).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from framework.graph_store import GraphStore


ARTIFACTS_ROOT = Path("artifacts")

# Experiment directories to materialize
EXPERIMENT_DIRS = [
    "rdf_vs_lpg",
    "rdf_vs_lpg_static",
    "rdf_vs_lpg_dynamic",
    "rdf_vs_lpg_none",
    "rdf_vs_lpg_rule_only",
    "rdf_vs_lpg_all_packs",
    # Legacy names (kept for backward compatibility with existing artifacts)
    "rdf_vs_lpg_smoke",
    "rdf_vs_lpg_smoke_static",
    "rdf_vs_lpg_smoke_dynamic",
    "rdf_vs_lpg_smoke_none",
    "rdf_vs_lpg_balanced",
    "rdf_vs_lpg_balanced_static",
    "rdf_vs_lpg_balanced_dynamic",
    "rdf_vs_lpg_balanced_none",
]


def load_artifact(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  SKIP {path}: {exc}")
        return None


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Re-materialize all artifacts into DozerDB.")
    parser.add_argument("--clear-first", action="store_true", help="Clear experiment data before loading")
    args = parser.parse_args()

    gs = GraphStore()
    if not gs.verify_connection():
        print("ERROR: DozerDB not reachable")
        sys.exit(1)

    gs.ensure_databases()
    gs.ensure_indexes()
    print("DozerDB connected, databases and indexes ready\n")

    total_rdf = 0
    total_lpg = 0
    errors = 0

    for exp_dir_name in EXPERIMENT_DIRS:
        exp_path = ARTIFACTS_ROOT / exp_dir_name
        if not exp_path.is_dir():
            continue

        print(f"=== {exp_dir_name} ===")

        if args.clear_first:
            gs.clear_experiment(exp_dir_name, database="finderrdf")
            gs.clear_experiment(exp_dir_name, database="finderlpg")
            print("  Cleared prior data")

        # RDF artifacts
        rdf_dir = exp_path / "rdf"
        if rdf_dir.is_dir():
            for artifact_path in sorted(rdf_dir.glob("*.json")):
                artifact = load_artifact(artifact_path)
                if artifact is None:
                    errors += 1
                    continue
                if artifact.get("status") != "completed":
                    continue
                try:
                    count = gs.materialize_rdf_as_lpg(artifact)
                    total_rdf += count
                except Exception as exc:
                    print(f"  RDF ERROR {artifact_path.name}: {exc}")
                    errors += 1

        # LPG artifacts
        lpg_dir = exp_path / "lpg"
        if lpg_dir.is_dir():
            for artifact_path in sorted(lpg_dir.glob("*.json")):
                artifact = load_artifact(artifact_path)
                if artifact is None:
                    errors += 1
                    continue
                if artifact.get("status") != "completed":
                    continue
                try:
                    count = gs.materialize_lpg(artifact)
                    total_lpg += count
                except Exception as exc:
                    print(f"  LPG ERROR {artifact_path.name}: {exc}")
                    errors += 1

        rdf_count = len(list(rdf_dir.glob("*.json"))) if rdf_dir.is_dir() else 0
        lpg_count = len(list(lpg_dir.glob("*.json"))) if lpg_dir.is_dir() else 0
        print(f"  rdf={rdf_count} lpg={lpg_count}")

    gs.close()

    print(f"\n{'=' * 50}")
    print(f"Total RDF elements materialized: {total_rdf}")
    print(f"Total LPG elements materialized: {total_lpg}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
