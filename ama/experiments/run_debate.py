"""
Run the RDF vs LPG debate pool on FinDER questions.

Usage:
    uv run python -m experiments.run_debate \
        --dataset-id Linq-AI-Research/FinDER \
        --sample-size 3

Prerequisites:
    1. One DozerDB instance running:
       docker compose up -d dozerdb

    2. Graphs populated from a prior extraction run:
       uv run python -m experiments.rdf_vs_lpg_balanced --sample-size 30 --materialize

Pipeline per question:

    load question → DebatePool.run():
        generate_independently (RDF + LPG agents query their graphs)
        → compare_outputs (agreement/disagreement analysis)
        → decide_strategy (direct_synthesis | self_reflection | debate | judge)
        → execute_strategy (critique round if needed)
        → synthesize (final answer with attribution)
    → save debate artifact JSON

Each debate run is traced via Opik for dashboard inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from debate.debate_pool import DebateExecutionError, DebatePool
from debate.graph_tools import clear_active_scope, close_connections, init_connections, set_active_scope
from framework.loader import balanced_sample, load_finder_dataset
from tracing.tracing import make_tracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RDF vs LPG debate on FinDER questions.")
    parser.add_argument("--dataset-id", default=os.getenv("FINDER_DATASET_ID", ""))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--category", action="append", default=[], help="Restrict to one or more categories; can be passed multiple times")
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--question-field", default="text")
    parser.add_argument("--id-field", default="_id")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--experiment-name", default="debate_rdf_vs_lpg")
    parser.add_argument("--source-experiment", default=os.getenv("DEBATE_SOURCE_EXPERIMENT", "rdf_vs_lpg_balanced"))
    parser.add_argument(
        "--retrieval-variant",
        choices=["search_a_v1", "search_a_v3", "search_a_v4", "search_a_v5"],
        default="search_a_v1",
        help="Retrieval/control variant for GraphRAG debate runs",
    )
    parser.add_argument("--no-cloud-trace", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.dataset_id:
        raise SystemExit("Missing dataset id. Set FINDER_DATASET_ID in .env or pass --dataset-id.")

    # Initialize graph connections
    print("Connecting to DozerDB...")
    init_connections()

    # Load questions from FinDER
    print(f"Loading dataset: {args.dataset_id}")
    examples = load_finder_dataset(args.dataset_id, args.split)
    if args.category:
        wanted = {value.strip().lower() for value in args.category if value.strip()}
        examples = [
            example for example in examples
            if str(example.get(args.category_field, "")).strip().lower() in wanted
        ]
        print(f"Filtered to categories: {', '.join(sorted(wanted))} -> {len(examples)} examples")
    sampled = balanced_sample(
        examples,
        sample_size=args.sample_size,
        category_field=args.category_field,
        random_seed=args.seed,
    )
    print(f"Sampled {len(sampled)} questions")

    # Create tracer
    tracer = make_tracer(
        project_name=os.getenv("OPIK_PROJECT_NAME", "ama_rdf_lpg_observability"),
        use_cloud=not args.no_cloud_trace,
    )

    # Create debate pool
    pool = DebatePool(model=args.model, tracer=tracer, retrieval_variant=args.retrieval_variant)

    output_dir = Path("artifacts") / args.experiment_name
    manifest: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "model": args.model,
        "retrieval_variant": args.retrieval_variant,
        "retrieval_scope_mode": "example_scoped",
        "reference_bound": True,
        "source_experiment": args.source_experiment,
        "sample_size": len(sampled),
        "debates": [],
    }

    for idx, example in enumerate(sampled):
        example_id = str(example.get(args.id_field, f"idx-{idx}"))
        category = str(example.get(args.category_field, "unknown"))
        question = str(example.get(args.question_field, ""))

        print(f"\n[{idx+1}/{len(sampled)}] {category} | {example_id}")
        print(f"  Q: {question[:120]}{'...' if len(question) > 120 else ''}")

        try:
            set_active_scope(experiment_name=args.source_experiment, example_id=example_id)
            result = pool.run(question, experiment_name=args.experiment_name)

            print(f"  Strategy: {result.handoff_log.final_resolution_mode}")
            print(f"  Rationale: {result.handoff_log.strategy_rationale}")
            print(f"  RDF confidence: {result.rdf_output.confidence:.2f} ({len(result.rdf_output.claims)} claims)")
            print(f"  LPG confidence: {result.lpg_output.confidence:.2f} ({len(result.lpg_output.claims)} claims)")
            print(f"  RDF tool digests: {len(result.rdf_output.tool_trace_digest)}")
            print(f"  LPG tool digests: {len(result.lpg_output.tool_trace_digest)}")
            print(f"  Agreement: {result.comparison.answers_agree}")
            print(f"  Final: {result.synthesis.selected_supporting_representation} "
                  f"(conf={result.synthesis.final_confidence:.2f})")
            print(f"  Answer: {result.synthesis.final_answer[:150]}{'...' if len(result.synthesis.final_answer) > 150 else ''}")

            # Save debate artifact
            artifact_path = output_dir / f"{example_id}.json"
            write_json(artifact_path, {
                "example_id": example_id,
                "category": category,
                "question": question,
                "representation_condition": result.representation_condition,
                "retrieval_variant": args.retrieval_variant,
                "source_experiment": args.source_experiment,
                "retrieval_scope_mode": "example_scoped",
                "reference_bound": True,
                "router_policy": {
                    "strategy": result.handoff_log.final_resolution_mode,
                    "strategy_rationale": result.handoff_log.strategy_rationale,
                    "decision_thresholds": result.handoff_log.decision_thresholds,
                },
                "tool_trace_counts": {
                    "rdf": len(result.rdf_output.tool_trace_digest),
                    "lpg": len(result.lpg_output.tool_trace_digest),
                },
                "retrieval_summary": {
                    "question_type": result.rdf_output.question_type or result.lpg_output.question_type,
                    "rdf_support_level": result.rdf_output.retrieval_support_level,
                    "lpg_support_level": result.lpg_output.retrieval_support_level,
                    "rdf_templates": result.rdf_output.retrieval_diagnostics.template_used,
                    "lpg_templates": result.lpg_output.retrieval_diagnostics.template_used,
                    "retrieval_scope_mode": "example_scoped",
                    "reference_bound": True,
                },
                "debate_result": result.model_dump(),
            })

            manifest["debates"].append({
                "example_id": example_id,
                "category": category,
                "representation_condition": result.representation_condition,
                "retrieval_variant": args.retrieval_variant,
                "source_experiment": args.source_experiment,
                "retrieval_scope_mode": "example_scoped",
                "reference_bound": True,
                "strategy": result.handoff_log.final_resolution_mode,
                "strategy_rationale": result.handoff_log.strategy_rationale,
                "decision_thresholds": result.handoff_log.decision_thresholds,
                "rdf_confidence": result.rdf_output.confidence,
                "lpg_confidence": result.lpg_output.confidence,
                "rdf_claim_count": len(result.rdf_output.claims),
                "lpg_claim_count": len(result.lpg_output.claims),
                "rdf_evidence_count": len(result.rdf_output.evidence_items),
                "lpg_evidence_count": len(result.lpg_output.evidence_items),
                "rdf_tool_trace_count": len(result.rdf_output.tool_trace_digest),
                "lpg_tool_trace_count": len(result.lpg_output.tool_trace_digest),
                "question_type": result.rdf_output.question_type or result.lpg_output.question_type,
                "rdf_support_level": result.rdf_output.retrieval_support_level,
                "lpg_support_level": result.lpg_output.retrieval_support_level,
                "rdf_template_used": result.rdf_output.retrieval_diagnostics.template_used,
                "lpg_template_used": result.lpg_output.retrieval_diagnostics.template_used,
                "rdf_freeform_query_used": result.rdf_output.retrieval_diagnostics.freeform_query_used,
                "lpg_freeform_query_used": result.lpg_output.retrieval_diagnostics.freeform_query_used,
                "answers_agree": result.comparison.answers_agree,
                "evidence_overlap_ratio": result.comparison.evidence_overlap_ratio,
                "confidence_gap": result.comparison.confidence_gap,
                "final_confidence": result.synthesis.final_confidence,
                "supporting_representation": result.synthesis.selected_supporting_representation,
                "resolution_mode": result.synthesis.resolution_mode,
                "artifact_path": str(artifact_path),
                "trace_id": result.trace_id,
                "opik_backend": result.trace_linkage.opik_backend,
                "openai_trace_stages": sorted(result.trace_linkage.openai_traces.keys()),
                "openai_trace_ids": {
                    stage: record.openai_trace_id
                    for stage, record in result.trace_linkage.openai_traces.items()
                },
            })

        except Exception as exc:
            print(f"  ERROR: {exc}")
            error_payload: dict[str, Any] = {
                "example_id": example_id,
                "category": category,
                "status": "error",
                "error": str(exc),
            }
            if isinstance(exc, DebateExecutionError):
                error_payload["error_detail"] = exc.to_dict()
                print(
                    "  Failure detail:",
                    f"stage={exc.stage_name}",
                    f"type={exc.exception_type}",
                    f"scope={exc.active_scope}",
                )
            else:
                error_payload["error_detail"] = {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            manifest["debates"].append(error_payload)
        finally:
            clear_active_scope()

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    # Summary
    debates = manifest["debates"]
    completed = [d for d in debates if "strategy" in d]
    errors = [d for d in debates if d.get("status") == "error"]

    print(f"\n{'='*60}")
    print(f"Debate Summary")
    print(f"{'='*60}")
    print(f"Total: {len(debates)}, Completed: {len(completed)}, Errors: {len(errors)}")

    if completed:
        strategies = [d["strategy"] for d in completed]
        for s in set(strategies):
            print(f"  {s}: {strategies.count(s)}")

        agree_count = sum(1 for d in completed if d["answers_agree"])
        print(f"  Agreement rate: {agree_count}/{len(completed)}")

        representations = [d.get("supporting_representation", "?") for d in completed]
        for r in set(representations):
            print(f"  Supported by {r}: {representations.count(r)}")

    print(f"\nManifest: {manifest_path}")

    close_connections()


if __name__ == "__main__":
    main()
