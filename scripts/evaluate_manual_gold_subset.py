#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tutorials.finance_graph_pipeline.framework.manual_gold import load_manual_gold


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _precision_recall_f1(predicted: set[Any], gold: set[Any]) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _artifact_root_from_args(args: argparse.Namespace, project_root: Path) -> Path:
    if args.artifact_root:
        return Path(args.artifact_root).resolve()
    if not args.run_id:
        raise SystemExit("Provide either --run-id or --artifact-root.")
    return (project_root / "data" / "experiment_runs" / "artifacts" / args.run_id).resolve()


def _selected_baselines(artifact_root: Path, include: list[str]) -> list[str]:
    available = sorted(path.name for path in artifact_root.iterdir() if path.is_dir())
    if not include:
        return available
    requested = []
    for baseline in include:
        if baseline in available:
            requested.append(baseline)
    return requested


def _bundle_text(answer_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    answer_text = str(answer_payload.get("answer", ""))
    if answer_text:
        parts.append(answer_text)

    bundle = answer_payload.get("evidence_bundle") or {}
    for sentence in bundle.get("reference_sentences", []):
        parts.append(str(sentence))
    for triple in bundle.get("triples", []):
        parts.extend(
            [
                str(triple.get("source_name", "")),
                str(triple.get("relation_type", "")),
                str(triple.get("target_name", "")),
            ]
        )
        for snippet in triple.get("provenance_snippets", []):
            parts.append(str(snippet))
    return _normalize(" ".join(parts))


def _evaluate_artifact(payload: dict[str, Any], gold: Any) -> dict[str, Any]:
    profile_decision = payload.get("profile_decision") or {}
    extraction = payload.get("extraction") or {}
    answer_payload = payload.get("answer") or {}
    query_support = payload.get("query_support") or {}
    evidence_bundle = answer_payload.get("evidence_bundle") or {}

    predicted_profile = profile_decision.get("selected_profile")
    predicted_intent = (
        query_support.get("intent_id")
        or evidence_bundle.get("intent_id")
        or ""
    )

    predicted_entities = {
        (_normalize(item.get("name", "")), item.get("entity_type"))
        for item in extraction.get("entities", [])
    }
    gold_entities = {
        (_normalize(name), entity_type)
        for name, entity_type in gold.gold_extraction.entities
    }

    predicted_relations = {
        (
            _normalize(item.get("source_name", "")),
            item.get("relation_type"),
            _normalize(item.get("target_name", "")),
        )
        for item in extraction.get("relations", [])
    }
    gold_relations = {
        (_normalize(source_name), relation_type, _normalize(target_name))
        for source_name, relation_type, target_name in gold.gold_extraction.relations
    }

    entity_metrics = _precision_recall_f1(predicted_entities, gold_entities)
    relation_metrics = _precision_recall_f1(predicted_relations, gold_relations)
    extraction_f1 = round((entity_metrics["f1"] + relation_metrics["f1"]) / 2, 3)

    required_slots = set(gold.required_answer_slots)
    filled_slots = set(evidence_bundle.get("filled_slots", []))
    slot_coverage = None
    if required_slots:
        slot_coverage = round(len(required_slots & filled_slots) / len(required_slots), 3)

    evidence_text = _bundle_text(answer_payload)
    preferred_terms = list(gold.preferred_answer_evidence)
    preferred_hit_rate = None
    if preferred_terms:
        hits = sum(1 for term in preferred_terms if _normalize(term) in evidence_text)
        preferred_hit_rate = round(hits / len(preferred_terms), 3)

    return {
        "example_id": gold.example_id,
        "category": gold.category,
        "gold_profile": gold.gold_profile,
        "predicted_profile": predicted_profile,
        "profile_match": predicted_profile == gold.gold_profile,
        "gold_intent_id": gold.intent_id,
        "predicted_intent_id": predicted_intent,
        "intent_match": predicted_intent == gold.intent_id,
        "entity_metrics": entity_metrics,
        "relation_metrics": relation_metrics,
        "extraction_f1_manual": extraction_f1,
        "required_answer_slots": list(gold.required_answer_slots),
        "filled_slots": sorted(filled_slots),
        "required_slot_coverage_manual": slot_coverage,
        "preferred_answer_evidence": preferred_terms,
        "preferred_evidence_hit_rate": preferred_hit_rate,
        "answer_token_f1": answer_payload.get("token_f1"),
        "evaluation_gold_source": extraction.get("evaluation_gold_source"),
        "artifact_error": payload.get("error"),
    }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FinDER run against the manual gold subset.")
    parser.add_argument("--run-id", help="Run id under data/experiment_runs/artifacts/")
    parser.add_argument("--artifact-root", help="Explicit artifact root directory.")
    parser.add_argument(
        "--manual-gold-path",
        default="tutorials/finance_graph_pipeline/data/finder_manual_gold_subset.json",
        help="Path to the manual gold subset JSON.",
    )
    parser.add_argument(
        "--include-baseline",
        action="append",
        default=[],
        help="Optional baseline directory name to evaluate. Repeat to include multiple baselines.",
    )
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    artifact_root = _artifact_root_from_args(args, PROJECT_ROOT)
    manual_gold_path = Path(args.manual_gold_path).resolve()
    manual_gold = load_manual_gold(manual_gold_path)

    if not artifact_root.exists():
        raise SystemExit(f"Artifact root not found: {artifact_root}")
    if not manual_gold:
        raise SystemExit(f"Manual gold subset is empty: {manual_gold_path}")

    baselines = _selected_baselines(artifact_root, args.include_baseline)
    if not baselines:
        raise SystemExit(f"No baselines found under {artifact_root}")

    payload: dict[str, Any] = {
        "run_id": args.run_id,
        "artifact_root": str(artifact_root),
        "manual_gold_path": str(manual_gold_path),
        "manual_gold_count": len(manual_gold),
        "baselines": {},
    }

    for baseline in baselines:
        baseline_root = artifact_root / baseline
        details: list[dict[str, Any]] = []
        missing_examples: list[str] = []

        for example_id, gold in manual_gold.items():
            artifact_path = baseline_root / f"{example_id}.json"
            if not artifact_path.exists():
                missing_examples.append(example_id)
                continue
            artifact_payload = json.loads(artifact_path.read_text())
            details.append(_evaluate_artifact(artifact_payload, gold))

        baseline_summary = {
            "sample_count": len(details),
            "missing_artifact_count": len(missing_examples),
            "profile_selection_accuracy_manual": _mean(
                [1.0 if item["profile_match"] else 0.0 for item in details]
            ),
            "intent_accuracy_manual": _mean(
                [1.0 if item["intent_match"] else 0.0 for item in details]
            ),
            "ontology_constrained_extraction_f1_manual": _mean(
                [item["extraction_f1_manual"] for item in details]
            ),
            "required_answer_slot_coverage_manual": _mean(
                [item["required_slot_coverage_manual"] for item in details]
            ),
            "preferred_answer_evidence_hit_rate": _mean(
                [item["preferred_evidence_hit_rate"] for item in details]
            ),
            "answer_token_f1": _mean([item["answer_token_f1"] for item in details]),
        }

        payload["baselines"][baseline] = {
            "summary": baseline_summary,
            "missing_examples": missing_examples,
            "per_example": details,
        }

    serialized = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
