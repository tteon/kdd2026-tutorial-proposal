from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from framework.schema import DivergenceSummary, ExampleChainView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an example-centric RDF/LPG chain walkthrough artifact.")
    parser.add_argument("--source-experiment", required=True, help="Indexing experiment directory under artifacts/")
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--debate-experiment", default="", help="Optional debate experiment directory under artifacts/")
    parser.add_argument("--output-path", default="", help="Optional explicit output path")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _preview_text(value: str, limit: int = 1200) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _build_divergence(rdf_artifact: dict[str, Any], lpg_artifact: dict[str, Any]) -> DivergenceSummary:
    rdf_links = rdf_artifact.get("linked_entities", [])
    lpg_links = lpg_artifact.get("linked_entities", [])

    rdf_names = {item.get("canonical_name", ""): item for item in rdf_links if item.get("canonical_name")}
    lpg_names = {item.get("canonical_name", ""): item for item in lpg_links if item.get("canonical_name")}

    shared = sorted(set(rdf_names) & set(lpg_names))
    rdf_only = sorted(set(rdf_names) - set(lpg_names))
    lpg_only = sorted(set(lpg_names) - set(rdf_names))

    differing: list[dict[str, Any]] = []
    for name in shared:
        rdf_target = rdf_names[name].get("target_id", "")
        lpg_target = lpg_names[name].get("target_id", "")
        if rdf_target != lpg_target:
            differing.append(
                {
                    "canonical_name": name,
                    "rdf_target_id": rdf_target,
                    "lpg_target_id": lpg_target,
                }
            )

    return DivergenceSummary(
        shared_canonical_names=shared,
        rdf_only_canonical_names=rdf_only,
        lpg_only_canonical_names=lpg_only,
        differing_link_targets=differing,
    )


def main() -> None:
    args = parse_args()

    base = Path("artifacts") / args.source_experiment
    rdf_path = base / "rdf" / f"{args.example_id}.json"
    lpg_path = base / "lpg" / f"{args.example_id}.json"

    if not rdf_path.exists():
        raise SystemExit(f"Missing RDF artifact: {rdf_path}")
    if not lpg_path.exists():
        raise SystemExit(f"Missing LPG artifact: {lpg_path}")

    rdf_artifact = _load_json(rdf_path)
    lpg_artifact = _load_json(lpg_path)

    debate_view: dict[str, Any] = {}
    if args.debate_experiment:
        debate_path = Path("artifacts") / args.debate_experiment / f"{args.example_id}.json"
        if debate_path.exists():
            debate_payload = _load_json(debate_path)
            result = debate_payload.get("debate_result", {})
            synthesis = result.get("synthesis", {})
            debate_view = {
                "experiment_name": args.debate_experiment,
                "strategy": debate_payload.get("router_policy", {}).get("strategy", ""),
                "strategy_rationale": debate_payload.get("router_policy", {}).get("strategy_rationale", ""),
                "final_answer": synthesis.get("final_answer", ""),
                "selected_supporting_representation": synthesis.get("selected_supporting_representation", ""),
                "resolution_mode": synthesis.get("resolution_mode", ""),
            }

    chain = ExampleChainView(
        example_id=args.example_id,
        category=rdf_artifact.get("category", ""),
        question=rdf_artifact.get("question", ""),
        references_preview=_preview_text(rdf_artifact.get("raw_input_text", "")),
        ontology={
            "selection_mode": rdf_artifact.get("ontology_selection", {}).get("selection_mode", ""),
            "selected_pack_keys": rdf_artifact.get("ontology_selection", {}).get("selected_pack_keys", []),
            "selection_reason": rdf_artifact.get("ontology_selection", {}).get("selection_reason", ""),
            "prompt_rule_labels": [
                rule.get("label", "")
                for rule in rdf_artifact.get("prompt_context", {}).get("rules", [])[:12]
            ],
        },
        rdf_view={
            "trace_id": rdf_artifact.get("trace_id", ""),
            "trace_backend": rdf_artifact.get("trace_backend", ""),
            "extracted_entities": rdf_artifact.get("extracted_entities", []),
            "linked_entities": rdf_artifact.get("linked_entities", []),
            "graph_preview": rdf_artifact.get("graph_preview", {}),
            "metrics": rdf_artifact.get("metrics", {}),
        },
        lpg_view={
            "trace_id": lpg_artifact.get("trace_id", ""),
            "trace_backend": lpg_artifact.get("trace_backend", ""),
            "extracted_entities": lpg_artifact.get("extracted_entities", []),
            "linked_entities": lpg_artifact.get("linked_entities", []),
            "graph_preview": lpg_artifact.get("graph_preview", {}),
            "metrics": lpg_artifact.get("metrics", {}),
        },
        divergence=_build_divergence(rdf_artifact, lpg_artifact),
        debate_view=debate_view,
    )

    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path("artifacts") / args.source_experiment / "chain_views" / f"{args.example_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chain.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
