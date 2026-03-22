"""
Batch ablation evaluation — run opik.evaluate() across multiple experiments.

Each experiment gets a distinct experiment_config in Opik, enabling
side-by-side comparison in the dashboard:

  ontology_mode (none / static / dynamic / all_packs / rule_only)
  × representation (rdf / lpg)
  × model (gpt-4o)

Usage:
    # Evaluate all existing ablation experiments
    uv run python -m experiments.eval_ablation

    # Evaluate specific experiments
    uv run python -m experiments.eval_ablation \
        --experiments rdf_vs_lpg_balanced_none rdf_vs_lpg_balanced_static

    # Also run local CLI evaluation (JSON + stdout)
    uv run python -m experiments.eval_ablation --local

    # Shared Opik dataset name (default: ama_ablation)
    uv run python -m experiments.eval_ablation --dataset-name my_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from framework.evaluation import run_opik_evaluate
from manual_gold.gold_schema import load_gold_examples


ARTIFACTS_ROOT = Path("artifacts")

# Ablation experiment directories in canonical order
ABLATION_EXPERIMENTS = [
    "rdf_vs_lpg",
    "rdf_vs_lpg_balanced_none",
    "rdf_vs_lpg_balanced_static",
    "rdf_vs_lpg_balanced_dynamic",
    "rdf_vs_lpg_balanced_all_packs",
    "rdf_vs_lpg_balanced_rule_only",
    # Pilot runs
    "rdf_vs_lpg_balanced_pilot10",
    "rdf_vs_lpg_balanced_pilot30",
    # Legacy names
    "rdf_vs_lpg_smoke_none",
    "rdf_vs_lpg_smoke_static",
    "rdf_vs_lpg_smoke_dynamic",
]


def find_experiments(requested: list[str] | None) -> list[str]:
    """Find experiment directories that have manifests."""
    if requested:
        candidates = requested
    else:
        candidates = ABLATION_EXPERIMENTS

    found = []
    for name in candidates:
        manifest = ARTIFACTS_ROOT / name / "manifest.json"
        if manifest.exists():
            found.append(name)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch ablation evaluation with Opik experiments.")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiment names (default: all found)")
    parser.add_argument("--dataset-name", default="ama_ablation",
                        help="Opik dataset name prefix (default: ama_ablation)")
    parser.add_argument("--local", action="store_true",
                        help="Also run local eval_smoke for each experiment")
    parser.add_argument("--project-name", default="ama_evaluation",
                        help="Opik project name (default: ama_evaluation)")
    args = parser.parse_args()

    experiments = find_experiments(args.experiments)
    if not experiments:
        print("No experiment manifests found.")
        return

    print(f"Found {len(experiments)} experiments to evaluate:")
    for name in experiments:
        print(f"  - {name}")

    # Load gold once
    gold_path = "manual_gold/seed_gold.json"
    gold_by_id: dict[str, Any] | None = None
    if Path(gold_path).exists():
        gold_examples = load_gold_examples(gold_path)
        gold_by_id = {g.example_id: g for g in gold_examples}
        print(f"\nGold: {len(gold_examples)} examples")

    print()

    results = []
    for name in experiments:
        manifest_path = ARTIFACTS_ROOT / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_examples = len(manifest.get("examples", []))
        ontology_mode = manifest.get("experiment", {}).get("ontology_mode", "unknown")
        model = manifest.get("model", "unknown")

        print(f"{'=' * 70}")
        print(f"Experiment: {name}")
        print(f"  ontology_mode={ontology_mode}  model={model}  examples={n_examples}")
        print(f"{'=' * 70}")

        try:
            result = run_opik_evaluate(
                experiment_name=name,
                manifest=manifest,
                gold_by_id=gold_by_id,
                project_name=args.project_name,
            )
            print(f"  Opik experiment: {result.experiment_name}")
            print(f"  URL: {result.experiment_url}")
            print(f"  Test results: {len(result.test_results)}")
            for es in result.experiment_scores:
                print(f"  Score: {es.name} = {es.value:.4f}")
            results.append({
                "experiment": name,
                "ontology_mode": ontology_mode,
                "opik_experiment_name": result.experiment_name,
                "opik_url": result.experiment_url,
                "n_results": len(result.test_results),
                "scores": {es.name: es.value for es in result.experiment_scores},
            })
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "experiment": name,
                "ontology_mode": ontology_mode,
                "error": str(exc),
            })

        print()

    # Summary table
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<40s} {'Mode':<12s} {'FIBO':>6s} {'F1':>6s}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['experiment']:<40s} {r['ontology_mode']:<12s}  ERROR")
            continue
        scores = r.get("scores", {})
        fibo = scores.get("macro_fibo_overall", 0)
        f1 = scores.get("macro_gold_f1", 0)
        print(f"{r['experiment']:<40s} {r['ontology_mode']:<12s} {fibo:6.4f} {f1:6.4f}")

    # Save summary
    summary_path = ARTIFACTS_ROOT / "ablation_evaluation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {summary_path}")

    # Optionally run local eval for each
    if args.local:
        import subprocess
        print(f"\n{'=' * 70}")
        print("Running local eval_smoke for each experiment...")
        for name in experiments:
            print(f"\n--- {name} ---")
            subprocess.run(
                ["uv", "run", "python", "-m", "experiments.eval_smoke",
                 "--experiment", name],
                check=False,
            )


if __name__ == "__main__":
    main()
