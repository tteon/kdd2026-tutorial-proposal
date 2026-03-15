#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tutorials.finance_graph_pipeline.framework.fibo_profiles import FIBO_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline-wise graph differences for a FinDER run.")
    parser.add_argument("--run-id", required=True, help="Experiment run id.")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Override artifact root. Defaults to data/experiment_runs/artifacts/<run-id>.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/graph_analysis",
        help="Directory to write JSON and CSV outputs.",
    )
    return parser.parse_args()


def _artifact_root(run_id: str, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return (PROJECT_ROOT / "data" / "experiment_runs" / "artifacts" / run_id).resolve()


def _load_summary(run_id: str) -> dict[str, Any] | None:
    summary_path = PROJECT_ROOT / "data" / "experiment_runs" / f"{run_id}_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    return None


def _baseline_order(run_id: str, artifact_root: Path) -> list[str]:
    summary = _load_summary(run_id)
    if summary:
        ordered = []
        ordered.extend(summary.get("auxiliary_baseline_order", []))
        ordered.extend(summary.get("ontology_baseline_order", []))
        if ordered:
            return ordered
    return sorted(path.name for path in artifact_root.iterdir() if path.is_dir())


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _normalize_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _load_artifacts_for_baseline(baseline_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for artifact_path in sorted(baseline_root.glob("*.json")):
        payload = json.loads(artifact_path.read_text())
        payload["_artifact_path"] = str(artifact_path)
        artifacts.append(payload)
    return artifacts


def _allowed_relation_ratio(relations: list[dict[str, Any]], profile: str | None) -> float | None:
    if not relations:
        return None
    if not profile or profile not in FIBO_PROFILES:
        return None
    allowed = set(FIBO_PROFILES[profile].relation_types)
    matched = sum(1 for relation in relations if relation.get("relation_type") in allowed)
    return round(matched / len(relations), 3)


def _artifact_record(payload: dict[str, Any]) -> dict[str, Any]:
    extraction = payload.get("extraction") or {}
    entities = extraction.get("entities", [])
    relations = extraction.get("relations", [])
    quality_issues = payload.get("quality_issues") or []
    query_support = payload.get("query_support") or {}
    answer = payload.get("answer") or {}
    evidence_bundle = answer.get("evidence_bundle") or {}
    profile_decision = payload.get("profile_decision") or {}
    selected_profile = profile_decision.get("selected_profile")

    issue_counter = Counter(issue.get("issue_type") for issue in quality_issues if issue.get("issue_type"))
    entity_types = [entity.get("entity_type") for entity in entities if entity.get("entity_type")]
    relation_types = [relation.get("relation_type") for relation in relations if relation.get("relation_type")]

    support_score = query_support.get("support_score")
    token_f1 = answer.get("token_f1")
    selected_edge_ids = answer.get("selected_edge_ids") or []
    filled_slots = query_support.get("filled_slots") or evidence_bundle.get("filled_slots") or []
    missing_requirements = query_support.get("missing_requirements") or evidence_bundle.get("missing_requirements") or []
    matched_relations = query_support.get("matched_relations") or evidence_bundle.get("matched_relations") or []
    matched_entity_types = query_support.get("matched_entity_types") or evidence_bundle.get("matched_entity_types") or []
    evidence_triples = evidence_bundle.get("triples") or []

    return {
        "example_id": payload.get("example_id"),
        "selected_profile": selected_profile,
        "entity_count": len(entities),
        "relation_count": len(relations),
        "unique_entity_type_count": len(set(entity_types)),
        "unique_relation_type_count": len(set(relation_types)),
        "allowed_relation_ratio": _allowed_relation_ratio(relations, selected_profile),
        "quality_issue_count": len(quality_issues),
        "schema_violation_count": issue_counter.get("schema_violation", 0),
        "disconnected_node_count": issue_counter.get("disconnected_node", 0),
        "dangling_edge_count": issue_counter.get("dangling_edge", 0),
        "support_score": float(support_score) if support_score is not None else None,
        "support_positive": bool(support_score and float(support_score) > 0.0),
        "answerable": bool(query_support.get("answerable")) if query_support else False,
        "matched_relation_count": len(matched_relations),
        "matched_entity_type_count": len(matched_entity_types),
        "filled_slot_count": len(filled_slots),
        "missing_requirement_count": len(missing_requirements),
        "selected_edge_count": len(selected_edge_ids),
        "evidence_triple_count": len(evidence_triples),
        "token_f1": float(token_f1) if token_f1 is not None else None,
        "entity_type_counter": Counter(entity_types),
        "relation_type_counter": Counter(relation_types),
        "issue_counter": issue_counter,
    }


def _aggregate_baseline(baseline: str, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    records = [_artifact_record(payload) for payload in payloads]
    entity_type_counter: Counter[str] = Counter()
    relation_type_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    profile_counter: Counter[str] = Counter()

    for record in records:
        entity_type_counter.update(record["entity_type_counter"])
        relation_type_counter.update(record["relation_type_counter"])
        issue_counter.update(record["issue_counter"])
        if record["selected_profile"]:
            profile_counter.update([record["selected_profile"]])

    summary = {
        "baseline": baseline,
        "sample_count": len(records),
        "selected_profiles": _normalize_counter(profile_counter),
        "avg_entity_count": _safe_mean([record["entity_count"] for record in records]),
        "avg_relation_count": _safe_mean([record["relation_count"] for record in records]),
        "avg_unique_entity_type_count": _safe_mean([record["unique_entity_type_count"] for record in records]),
        "avg_unique_relation_type_count": _safe_mean([record["unique_relation_type_count"] for record in records]),
        "relation_schema_conformance_rate": _safe_mean(
            [record["allowed_relation_ratio"] for record in records if record["allowed_relation_ratio"] is not None]
        ),
        "avg_quality_issue_count": _safe_mean([record["quality_issue_count"] for record in records]),
        "avg_schema_violation_count": _safe_mean([record["schema_violation_count"] for record in records]),
        "avg_disconnected_node_count": _safe_mean([record["disconnected_node_count"] for record in records]),
        "avg_dangling_edge_count": _safe_mean([record["dangling_edge_count"] for record in records]),
        "query_support_positive_rate": _safe_mean([1.0 if record["support_positive"] else 0.0 for record in records]),
        "answerable_rate": _safe_mean([1.0 if record["answerable"] else 0.0 for record in records]),
        "avg_support_score": _safe_mean(
            [record["support_score"] for record in records if record["support_score"] is not None]
        ),
        "avg_matched_relation_count": _safe_mean([record["matched_relation_count"] for record in records]),
        "avg_matched_entity_type_count": _safe_mean([record["matched_entity_type_count"] for record in records]),
        "avg_filled_slot_count": _safe_mean([record["filled_slot_count"] for record in records]),
        "avg_missing_requirement_count": _safe_mean([record["missing_requirement_count"] for record in records]),
        "avg_selected_edge_count": _safe_mean([record["selected_edge_count"] for record in records]),
        "avg_evidence_triple_count": _safe_mean([record["evidence_triple_count"] for record in records]),
        "avg_answer_token_f1": _safe_mean([record["token_f1"] for record in records if record["token_f1"] is not None]),
        "entity_type_counts": _normalize_counter(entity_type_counter),
        "relation_type_counts": _normalize_counter(relation_type_counter),
        "quality_issue_counts": _normalize_counter(issue_counter),
        "per_example": [
            {
                key: value
                for key, value in record.items()
                if key not in {"entity_type_counter", "relation_type_counter", "issue_counter"}
            }
            for record in records
        ],
    }
    return summary


def _delta_table(summaries: list[dict[str, Any]], baseline_name: str) -> list[dict[str, Any]]:
    baseline_map = {summary["baseline"]: summary for summary in summaries}
    reference = baseline_map.get(baseline_name)
    if reference is None:
        return []
    metric_names = [
        "avg_entity_count",
        "avg_relation_count",
        "relation_schema_conformance_rate",
        "avg_schema_violation_count",
        "avg_disconnected_node_count",
        "avg_support_score",
        "answerable_rate",
        "avg_filled_slot_count",
        "avg_selected_edge_count",
        "avg_answer_token_f1",
    ]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = {"baseline": summary["baseline"], "delta_against": baseline_name}
        for metric_name in metric_names:
            lhs = summary.get(metric_name)
            rhs = reference.get(metric_name)
            if lhs is None or rhs is None:
                row[f"{metric_name}_delta"] = None
            else:
                row[f"{metric_name}_delta"] = round(lhs - rhs, 3)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    artifact_root = _artifact_root(args.run_id, args.artifact_root)
    if not artifact_root.exists():
        raise SystemExit(f"Artifact root not found: {artifact_root}")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    baseline_order = _baseline_order(args.run_id, artifact_root)
    for baseline in baseline_order:
        baseline_root = artifact_root / baseline
        if not baseline_root.is_dir():
            continue
        payloads = _load_artifacts_for_baseline(baseline_root)
        if not payloads:
            continue
        summaries.append(_aggregate_baseline(baseline, payloads))

    delta_rows = _delta_table(summaries, "graph_without_ontology_constraints")

    payload = {
        "run_id": args.run_id,
        "artifact_root": str(artifact_root),
        "baseline_order": baseline_order,
        "baseline_summaries": summaries,
        "delta_against_graph_without_ontology_constraints": delta_rows,
    }

    json_path = output_dir / f"{args.run_id}_graph_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2))

    overview_rows = []
    for summary in summaries:
        overview_rows.append(
            {
                key: value
                for key, value in summary.items()
                if key not in {"entity_type_counts", "relation_type_counts", "quality_issue_counts", "per_example", "selected_profiles"}
            }
        )
    _write_csv(output_dir / f"{args.run_id}_graph_overview.csv", overview_rows)
    _write_csv(output_dir / f"{args.run_id}_graph_deltas.csv", delta_rows)

    entity_rows = []
    relation_rows = []
    issue_rows = []
    profile_rows = []
    per_example_rows = []
    for summary in summaries:
        baseline = summary["baseline"]
        for entity_type, count in summary["entity_type_counts"].items():
            entity_rows.append({"baseline": baseline, "entity_type": entity_type, "count": count})
        for relation_type, count in summary["relation_type_counts"].items():
            relation_rows.append({"baseline": baseline, "relation_type": relation_type, "count": count})
        for issue_type, count in summary["quality_issue_counts"].items():
            issue_rows.append({"baseline": baseline, "issue_type": issue_type, "count": count})
        for profile, count in summary["selected_profiles"].items():
            profile_rows.append({"baseline": baseline, "selected_profile": profile, "count": count})
        for record in summary["per_example"]:
            per_example_rows.append({"baseline": baseline, **record})

    _write_csv(output_dir / f"{args.run_id}_graph_entity_types.csv", entity_rows)
    _write_csv(output_dir / f"{args.run_id}_graph_relation_types.csv", relation_rows)
    _write_csv(output_dir / f"{args.run_id}_graph_quality_issues.csv", issue_rows)
    _write_csv(output_dir / f"{args.run_id}_graph_profiles.csv", profile_rows)
    _write_csv(output_dir / f"{args.run_id}_graph_per_example.csv", per_example_rows)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
