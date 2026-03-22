"""
RDF vs LPG indexing experiment — balanced category sampling.

Usage:
    # Quick smoke test
    uv run python -m experiments.run_indexing \
        --dataset-id Linq-AI-Research/FinDER --sample-size 4

    # Full run with DozerDB materialization
    uv run python -m experiments.run_indexing \
        --dataset-id Linq-AI-Research/FinDER --sample-size 30 --materialize

Runs the extraction pipeline for each sampled example × {rdf, lpg}:

    make_base_artifact → join_references → extract_entities
    → normalize_entities → link_entities → extract_relations
    → materialize_graph → compute_basic_metrics

Features:
  - Balanced category sampling (equal examples per category)
  - Optional --materialize flag to write into DozerDB (finderrdf/finderlpg)
  - RDF vs LPG comparison summary at the end
  - Opik Cloud tracing with token usage, FIBO conformance feedback
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from framework.analysis import rdf_lpg_comparison_summary
from framework.evaluation import evaluate_fibo_conformance
from framework.llm_client import get_openai_client
from framework.loader import balanced_sample, load_finder_dataset
from framework.pipeline import artifact_summary, prompt_trace_summary, run_extraction_pipeline
from framework.rdf_export import build_rdf_export_metadata, write_ttl_artifact
from framework.schema import ExperimentConfig
from tracing.tracing import make_tracer


def _trace_run_kind(experiment_name: str) -> str:
    lower = experiment_name.lower()
    if "balanced" in lower:
        return "balanced"
    if "smoke" in lower:
        return "smoke"
    if "pilot" in lower:
        return "pilot"
    return "run"


def _trace_name(*, experiment_name: str, representation: str, category: str, example_id: str) -> str:
    run_kind = _trace_run_kind(experiment_name)
    category_slug = category.strip().lower().replace(" ", "_")
    return f"indexing/{run_kind}/{representation}/{category_slug}/{example_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDF vs LPG indexing experiment.")
    parser.add_argument("--dataset-id", default=os.getenv("FINDER_DATASET_ID", ""))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--experiment-name", default="rdf_vs_lpg")
    parser.add_argument("--experiment-version", default="v1")
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--question-field", default="text")
    parser.add_argument("--document-field", default="references")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--id-field", default="_id")
    parser.add_argument("--ontology-mode", choices=["none", "static", "dynamic", "rule_only", "all_packs"], default="all_packs")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--materialize", action="store_true", help="Write results to DozerDB")
    parser.add_argument("--no-cloud-trace", action="store_true", help="Use local trace only")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.dataset_id:
        raise SystemExit("Missing dataset id. Set FINDER_DATASET_ID in .env or pass --dataset-id.")

    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_version=args.experiment_version,
        dataset_id=args.dataset_id,
        dataset_split=args.split,
        sample_size=args.sample_size,
        random_seed=args.seed,
        category_field=args.category_field,
        question_field=args.question_field,
        document_field=args.document_field,
        answer_field=args.answer_field,
        id_field=args.id_field,
        ontology_mode=args.ontology_mode,
        opik_project_name=os.getenv("OPIK_PROJECT_NAME", "ama_rdf_lpg_observability"),
    )

    print(f"Loading dataset: {config.dataset_id}")
    examples = load_finder_dataset(config.dataset_id, config.dataset_split)
    sampled = balanced_sample(
        examples,
        sample_size=config.sample_size,
        category_field=config.category_field,
        random_seed=config.random_seed,
    )
    print(f"Sampled {len(sampled)} examples (target: {config.sample_size})")

    # Show category distribution
    cat_counts: dict[str, int] = defaultdict(int)
    for ex in sampled:
        cat_counts[str(ex.get(config.category_field, "unknown"))] += 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    openai_client = get_openai_client()
    model = args.model

    # Optional DozerDB materialization
    graph_store = None
    if args.materialize:
        from framework.graph_store import GraphStore
        graph_store = GraphStore()
        try:
            if not graph_store.verify_connection():
                raise RuntimeError(
                    "DozerDB is not reachable, but --materialize was requested. "
                    "Start DozerDB first or rerun without --materialize."
                )
            print("DozerDB connected")
            graph_store.ensure_databases()
            graph_store.ensure_indexes()
            rdf_database = os.getenv("NEO4J_DATABASE_RDF", "finderrdf")
            lpg_database = os.getenv("NEO4J_DATABASE_LPG", "finderlpg")
            print(f"  RDF database: {rdf_database}")
            print(f"  LPG database: {lpg_database}")
            graph_store.clear_experiment(config.experiment_name, database=rdf_database)
            graph_store.clear_experiment(config.experiment_name, database=lpg_database)
        except Exception:
            graph_store.close()
            raise

    run_id = uuid4().hex[:12]  # groups all traces from this invocation
    print(f"Run ID: {run_id}")

    rdf_artifacts: list[dict[str, Any]] = []
    lpg_artifacts: list[dict[str, Any]] = []

    manifest: dict[str, Any] = {
        "experiment": asdict(config),
        "model": model,
        "materialize": args.materialize,
        "materialize_target": {
            "uri_env": "NEO4J_URI",
            "rdf_database": os.getenv("NEO4J_DATABASE_RDF", "finderrdf"),
            "lpg_database": os.getenv("NEO4J_DATABASE_LPG", "finderlpg"),
        },
        "sampled_example_count": len(sampled),
        "examples": [],
    }

    for idx, example in enumerate(sampled):
        example_id = str(example.get(config.id_field, f"idx-{idx}"))
        category = str(example.get(config.category_field, "unknown"))

        for representation in ("rdf", "lpg"):
            print(f"  [{idx+1}/{len(sampled)}] {representation.upper()} | {category} | {example_id}")

            tracer = make_tracer(
                project_name=config.opik_project_name,
                use_cloud=not args.no_cloud_trace,
            )
            trace_backend = getattr(tracer, "backend", "unknown")
            print(f"    tracing={trace_backend}")
            trace_id = tracer.begin_trace(
                name=_trace_name(
                    experiment_name=config.experiment_name,
                    representation=representation,
                    category=category,
                    example_id=example_id,
                ),
                metadata={
                    # Identity — unique trace identity
                    "example_id": example_id,
                    "representation": representation,
                    "category": category,
                    # Experiment — which run this belongs to
                    "experiment_name": config.experiment_name,
                    "experiment_version": config.experiment_version,
                    "run_kind": _trace_run_kind(config.experiment_name),
                    "run_id": run_id,
                    "pipeline_family": "indexing",
                    # Configuration — pipeline settings
                    "ontology_mode": config.ontology_mode,
                    "model": model,
                    "prompt_template_id": config.prompt_template_id,
                    # System
                    "trace_backend": trace_backend,
                    "dataset_id": config.dataset_id,
                    "dataset_split": config.dataset_split,
                },
                tags=[
                    _trace_run_kind(config.experiment_name),
                    representation,
                    config.ontology_mode,
                    category.strip().lower().replace(" ", "_"),
                ],
            )

            try:
                artifact = run_extraction_pipeline(
                    config=config,
                    example=example,
                    representation=representation,
                    tracer=tracer,
                    openai_client=openai_client,
                    model=model,
                )
                artifact.trace_id = trace_id
                artifact.trace_backend = trace_backend

                # Feedback scores — self-consistency metrics
                if artifact.metrics.null_link_rate is not None:
                    tracer.log_score("null_link_rate", artifact.metrics.null_link_rate)
                if artifact.metrics.materialization_success_rate is not None:
                    tracer.log_score("materialization_success_rate", artifact.metrics.materialization_success_rate)
                if artifact.metrics.duplicate_entity_rate is not None:
                    tracer.log_score("duplicate_entity_rate", artifact.metrics.duplicate_entity_rate)

                # Feedback scores — FIBO conformance
                try:
                    conformance = evaluate_fibo_conformance(artifact.to_dict())
                    if conformance.semantic_conformance is not None:
                        tracer.log_score("semantic_conformance", conformance.semantic_conformance)
                    if conformance.structural_conformance is not None:
                        import math
                        if not math.isnan(conformance.structural_conformance):
                            tracer.log_score("structural_conformance", conformance.structural_conformance)
                    tracer.log_score("fibo_namespace_rate", conformance.fibo_namespace_rate)
                    tracer.log_score("overall_conformance", conformance.overall_conformance)
                except Exception:
                    pass  # conformance eval failure should not block pipeline

            except Exception as exc:
                print(f"    ERROR: {exc}")
                from framework.pipeline import make_base_artifact
                artifact = make_base_artifact(config=config, example=example, representation=representation)
                artifact.trace_id = trace_id
                artifact.trace_backend = trace_backend
                artifact.status = "error"
                artifact.error_notes.append(str(exc))

            tracer.end_trace(
                output={
                    "status": artifact.status,
                    "representation": artifact.representation,
                    "example_id": artifact.example_id,
                    "category": artifact.category,
                    "ontology_mode": artifact.ontology_selection.selection_mode,
                    "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
                    "entity_count": len(artifact.extracted_entities),
                    "linked_entity_count": len(artifact.linked_entities),
                    "error_notes": artifact.error_notes[:5],
                    "prompt_summary": prompt_trace_summary(artifact),
                },
                metadata={
                    "trace_backend": trace_backend,
                    "pipeline_family": "indexing",
                    "run_kind": _trace_run_kind(config.experiment_name),
                    "ontology_mode": config.ontology_mode,
                    "prompt_template_id": config.prompt_template_id,
                },
            )

            # Persist artifact
            artifact_dict = artifact.to_dict()
            artifact_path = (
                Path(config.output_dir) / config.experiment_name
                / representation / f"{artifact.example_id}.json"
            )
            write_json(artifact_path, artifact_dict)

            ttl_path = ""
            if representation == "rdf" and artifact.status == "completed":
                ttl_output_path = (
                    Path(config.output_dir) / config.experiment_name / "rdf_ttl" / f"{artifact.example_id}.ttl"
                )
                write_ttl_artifact(artifact_dict, output_path=ttl_output_path)
                artifact.rdf_export = build_rdf_export_metadata(artifact_dict, output_path=ttl_output_path)
                artifact_dict = artifact.to_dict()
                write_json(artifact_path, artifact_dict)
                ttl_path = str(ttl_output_path)

            # Collect for comparison
            if representation == "rdf":
                rdf_artifacts.append(artifact_dict)
            else:
                lpg_artifacts.append(artifact_dict)

            # Optional DozerDB materialization
            if graph_store and artifact.status == "completed":
                try:
                    if representation == "rdf":
                        graph_store.materialize_rdf_as_lpg(artifact_dict)
                    else:
                        graph_store.materialize_lpg(artifact_dict)
                except Exception as exc:
                    print(f"    Graph store error: {exc}")

            summary = artifact_summary(artifact)
            print(f"    → {summary['status']} | entities={summary['entity_count']} | linked={summary['linked_entity_count']}")

            manifest["examples"].append({
                "representation": representation,
                "example_id": artifact.example_id,
                "category": artifact.category,
                "status": artifact.status,
                "artifact_path": str(artifact_path),
                "trace_id": artifact.trace_id,
                "trace_backend": artifact.trace_backend,
                "ontology_mode": artifact.ontology_selection.selection_mode,
                "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
                "ttl_path": ttl_path,
            })

    if graph_store:
        graph_store.close()

    # Write manifest
    manifest_path = Path(config.output_dir) / config.experiment_name / "manifest.json"
    write_json(manifest_path, manifest)

    # Print comparison summary
    print("\n" + "=" * 60)
    print("RDF vs LPG Comparison Summary")
    print("=" * 60)
    comparison = rdf_lpg_comparison_summary(rdf_artifacts, lpg_artifacts)
    print(json.dumps(comparison, indent=2))

    # Write comparison to file
    comparison_path = Path(config.output_dir) / config.experiment_name / "comparison_summary.json"
    write_json(comparison_path, comparison)
    print(f"\nComparison saved: {comparison_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
