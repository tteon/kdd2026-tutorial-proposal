#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tutorials.finance_graph_pipeline.framework.fibo_profiles import FIBO_PROFILES, SECTION_TO_PROFILE

BASELINE_ORDER = (
    "question_only_baseline",
    "reference_only_baseline",
    "graph_without_ontology_constraints",
    "graph_with_profile_selection_only",
    "graph_with_profile_plus_constrained_extraction_and_linking",
    "full_minimal_pipeline_with_quality_aware_evidence_selection",
)

GRAPH_BASELINES = {
    "graph_without_ontology_constraints",
    "graph_with_profile_selection_only",
    "graph_with_profile_plus_constrained_extraction_and_linking",
    "full_minimal_pipeline_with_quality_aware_evidence_selection",
}

PROFILE_BASELINES = {
    "graph_with_profile_selection_only",
    "graph_with_profile_plus_constrained_extraction_and_linking",
    "full_minimal_pipeline_with_quality_aware_evidence_selection",
}

FINANCIAL_METRICS = (
    "revenue",
    "net income",
    "operating income",
    "operating margin",
    "gross profit",
    "cash",
    "cash equivalents",
    "liquidity",
    "debt",
    "cash flow",
    "earnings",
)

MONEY_PATTERN = re.compile(r"(?:USD\s*)?\$?\s?\d+(?:\.\d+)?\s?(?:billion|million|thousand|%)?", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:FY)?20\d{2}\b")
QUARTER_PATTERN = re.compile(r"\bQ[1-4]\s?20\d{2}\b", re.IGNORECASE)
PERSON_PATTERN = re.compile(r"\b([A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a proposal-oriented metrics report for FinDER runs.")
    parser.add_argument("--run-id", required=True, help="Primary FinDER run id.")
    parser.add_argument(
        "--db-path",
        default="data/metadata/experiment.sqlite",
        help="Path to the experiment SQLite database.",
    )
    parser.add_argument(
        "--manual-eval-json",
        default=None,
        help="Optional manual-gold evaluation JSON path.",
    )
    parser.add_argument(
        "--manual-eval-run-id",
        default=None,
        help="Optional run id whose exports/manual_gold/<run_id>_manual_gold_eval.json should be loaded.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/proposal_metrics",
        help="Directory for the generated report files.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser.parse_args()


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _precision_recall_f1(predicted: set[Any], gold: set[Any]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bootstrap_mean_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    mean = sum(values) / len(values)
    if len(values) == 1:
        rounded = round(mean, 3)
        return rounded, rounded, rounded
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    low_index = int(0.025 * len(boot))
    high_index = int(0.975 * len(boot))
    return round(mean, 3), round(boot[low_index], 3), round(boot[min(high_index, len(boot) - 1)], 3)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _manual_eval_path(args: argparse.Namespace) -> Path | None:
    if args.manual_eval_json:
        return Path(args.manual_eval_json).resolve()
    if args.manual_eval_run_id:
        candidate = PROJECT_ROOT / "exports" / "manual_gold" / f"{args.manual_eval_run_id}_manual_gold_eval.json"
        if candidate.is_file():
            return candidate
    candidate = PROJECT_ROOT / "exports" / "manual_gold" / f"{args.run_id}_manual_gold_eval.json"
    if candidate.is_file():
        return candidate
    return None


def _summary_path(run_id: str) -> Path | None:
    candidate = PROJECT_ROOT / "data" / "experiment_runs" / f"{run_id}_summary.json"
    if candidate.is_file():
        return candidate
    return None


def _load_run_summary(run_id: str) -> dict[str, Any] | None:
    path = _summary_path(run_id)
    if path is None:
        return None
    return json.loads(path.read_text())


def _load_answers_by_baseline(connection: sqlite3.Connection, run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    rows = connection.execute(
        """
        SELECT doc_id, category, evaluation_json
        FROM question_answers
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    answers: dict[str, dict[str, dict[str, Any]]] = {}
    for doc_id, category, evaluation_json in rows:
        evaluation = json.loads(evaluation_json) if evaluation_json else {}
        baseline = evaluation.get("baseline")
        if baseline is None:
            continue
        answers.setdefault(baseline, {})[doc_id] = {
            "doc_id": doc_id,
            "category": category,
            "token_f1": float(evaluation.get("token_f1", 0.0)),
        }
    return answers


def _load_artifacts(run_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    artifact_root = PROJECT_ROOT / "data" / "experiment_runs" / "artifacts" / run_id
    artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    if not artifact_root.exists():
        return artifacts
    for baseline_root in artifact_root.iterdir():
        if not baseline_root.is_dir():
            continue
        baseline_payloads: dict[str, dict[str, Any]] = {}
        for path in baseline_root.glob("*.json"):
            baseline_payloads[path.stem] = json.loads(path.read_text())
        artifacts[baseline_root.name] = baseline_payloads
    return artifacts


def _canonical_profile(category: str) -> str:
    return SECTION_TO_PROFILE[" ".join(category.lower().split())]


def _induced_gold(payload: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    answer = payload.get("answer") or {}
    text = f"{answer.get('question', '')}\n{answer.get('ground_truth_answer', '')}"
    category = str(answer.get("category", "")).lower()
    profile = _canonical_profile(category) if category else ""

    entities: list[tuple[str, str]] = []
    relations: list[tuple[str, str, str]] = []

    if profile == "governance":
        seen_people: set[str] = set()
        for match in PERSON_PATTERN.findall(text):
            normalized = " ".join(match.split())
            key = normalized.lower()
            if key in seen_people:
                continue
            seen_people.add(key)
            entities.append((_normalize(normalized), "Person"))
        lower = text.lower()
        if "audit committee" in lower:
            entities.append((_normalize("Audit Committee"), "Committee"))
            for name, entity_type in entities:
                if entity_type == "Person":
                    relations.append((name, "serves_on_committee", _normalize("Audit Committee")))
        if "chief executive officer" in lower:
            entities.append((_normalize("Chief Executive Officer"), "OfficerRole"))

    if profile == "financials":
        seen_amounts: set[str] = set()
        for match in MONEY_PATTERN.findall(text):
            normalized = " ".join(match.replace("\n", " ").split())
            if len(normalized) < 2:
                continue
            lowered = normalized.lower()
            if lowered in seen_amounts:
                continue
            seen_amounts.add(lowered)
            entities.append((_normalize(normalized), "MonetaryAmount"))
        periods = sorted(set(YEAR_PATTERN.findall(text) + QUARTER_PATTERN.findall(text)))
        for period in periods:
            entities.append((_normalize(period), "ReportingPeriod"))
        lower = text.lower()
        for metric in FINANCIAL_METRICS:
            if metric in lower:
                metric_name = _normalize(metric)
                entities.append((metric_name, "FinancialMetric"))
                if periods:
                    period = next(name for name, entity_type in entities if entity_type == "ReportingPeriod")
                    relations.append((metric_name, "reported_for_period", period))
                break

    if profile == "shareholder_return":
        lower = text.lower()
        if "repurchase" in lower or "buyback" in lower:
            entities.append((_normalize("Share Repurchase Program"), "RepurchaseProgram"))
            relations.append((_normalize("company"), "announced_repurchase", _normalize("Share Repurchase Program")))
        if "dividend" in lower:
            entities.append((_normalize("Dividend"), "Dividend"))
            relations.append((_normalize("company"), "declared_dividend", _normalize("Dividend")))
        if "common stock" in lower or "common share" in lower:
            share_class = "Common Stock" if "common stock" in lower else "Common Share"
            entities.append((_normalize(share_class), "ShareClass"))

    gold_entities = set(dict.fromkeys(entities))
    gold_relations = set(dict.fromkeys(relations))
    return gold_entities, gold_relations


def _artifact_profile_accuracy(payload: dict[str, Any]) -> float | None:
    answer = payload.get("answer") or {}
    category = answer.get("category")
    if not category:
        return None
    selected_profile = (payload.get("profile_decision") or {}).get("selected_profile")
    if not selected_profile:
        return None
    return 1.0 if selected_profile == _canonical_profile(category) else 0.0


def _artifact_extraction_f1(payload: dict[str, Any]) -> float | None:
    extraction = payload.get("extraction") or {}
    entities = extraction.get("entities") or []
    relations = extraction.get("relations") or []
    if not entities and not relations:
        return 0.0
    predicted_entities = {
        (_normalize(item.get("name", "")), item.get("entity_type"))
        for item in entities
    }
    predicted_relations = {
        (
            _normalize(item.get("source_name", "")),
            item.get("relation_type"),
            _normalize(item.get("target_name", "")),
        )
        for item in relations
    }
    gold_entities, gold_relations = _induced_gold(payload)
    entity_f1 = _precision_recall_f1(predicted_entities, gold_entities)
    relation_f1 = _precision_recall_f1(predicted_relations, gold_relations)
    return round((entity_f1 + relation_f1) / 2, 3)


def _artifact_support_score(payload: dict[str, Any]) -> float | None:
    query_support = payload.get("query_support")
    if not query_support:
        return None
    value = query_support.get("support_score")
    return None if value is None else float(value)


def _artifact_schema_conformance(payload: dict[str, Any]) -> float | None:
    extraction = payload.get("extraction") or {}
    relations = extraction.get("relations") or []
    if not relations:
        return None
    selected_profile = (payload.get("profile_decision") or {}).get("selected_profile")
    if selected_profile not in FIBO_PROFILES:
        return None
    allowed = set(FIBO_PROFILES[selected_profile].relation_types)
    matched = sum(1 for relation in relations if relation.get("relation_type") in allowed)
    return round(matched / len(relations), 3)


def _artifact_any_fallback_or_error(payload: dict[str, Any]) -> float | None:
    extraction = payload.get("extraction") or {}
    extraction_mode = (extraction.get("metadata") or {}).get("mode")
    evidence_mode = ((payload.get("answer") or {}).get("evidence_bundle") or {}).get("mode")
    rationale = ((payload.get("profile_decision") or {}).get("ontology_rationale") or "")
    if payload.get("error"):
        return 1.0
    if extraction_mode == "heuristic_fallback":
        return 1.0
    if evidence_mode == "runtime_error_fallback":
        return 1.0
    if "Heuristic fallback used" in rationale:
        return 1.0
    return 0.0


def _collect_primary_metric_values(
    run_id: str,
    answers_by_baseline: dict[str, dict[str, dict[str, Any]]],
    artifacts_by_baseline: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    reference_answers = answers_by_baseline.get("reference_only_baseline", {})

    for baseline in BASELINE_ORDER:
        baseline_answers = answers_by_baseline.get(baseline, {})
        baseline_artifacts = artifacts_by_baseline.get(baseline, {})
        doc_ids = sorted(set(baseline_answers) | set(baseline_artifacts))
        for doc_id in doc_ids:
            answer_row = baseline_answers.get(doc_id)
            artifact = baseline_artifacts.get(doc_id)
            category = None
            if answer_row is not None:
                category = answer_row.get("category")
            elif artifact is not None:
                category = (artifact.get("answer") or {}).get("category")
            if category is None:
                continue
            reference_answer = reference_answers.get(doc_id)
            answer_quality_score = answer_row.get("token_f1") if answer_row else None
            answer_quality_delta = None
            if answer_quality_score is not None and reference_answer is not None:
                answer_quality_delta = round(answer_quality_score - float(reference_answer["token_f1"]), 3)

            row = {
                "run_id": run_id,
                "baseline": baseline,
                "doc_id": doc_id,
                "category": category,
                "answer_quality_score": answer_quality_score,
                "answer_quality_delta": answer_quality_delta,
                "profile_selection_accuracy": None,
                "ontology_constrained_extraction_f1": None,
                "query_support_path_coverage": None,
                "relation_schema_conformance_rate": None,
                "fallback_error_rate": None,
            }
            if artifact is not None and baseline in GRAPH_BASELINES:
                row["ontology_constrained_extraction_f1"] = _artifact_extraction_f1(artifact)
                row["query_support_path_coverage"] = _artifact_support_score(artifact)
                row["relation_schema_conformance_rate"] = _artifact_schema_conformance(artifact)
                row["fallback_error_rate"] = _artifact_any_fallback_or_error(artifact)
            if artifact is not None and baseline in PROFILE_BASELINES:
                row["profile_selection_accuracy"] = _artifact_profile_accuracy(artifact)
            rows.append(row)
    return {"rows": rows}


def _summarize_metric_rows(
    rows: list[dict[str, Any]],
    metrics: list[str],
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for baseline in BASELINE_ORDER:
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        if not baseline_rows:
            continue
        for category in ["overall"] + sorted({row["category"] for row in baseline_rows}):
            category_rows = baseline_rows if category == "overall" else [row for row in baseline_rows if row["category"] == category]
            for metric in metrics:
                values = [float(row[metric]) for row in category_rows if row.get(metric) is not None]
                if not values:
                    continue
                mean, ci_low, ci_high = _bootstrap_mean_ci(values, resamples, seed)
                summary_rows.append(
                    {
                        "baseline": baseline,
                        "category": category,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n": len(values),
                    }
                )
    return summary_rows


def _load_manual_eval(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


def _manual_rows(manual_eval: dict[str, Any] | None) -> list[dict[str, Any]]:
    if manual_eval is None:
        return []
    rows: list[dict[str, Any]] = []
    for baseline, payload in manual_eval.get("baselines", {}).items():
        for item in payload.get("per_example", []):
            rows.append(
                {
                    "baseline": baseline,
                    "category": item["category"],
                    "required_answer_slot_coverage_manual": item.get("required_slot_coverage_manual"),
                    "preferred_answer_evidence_hit_rate": item.get("preferred_evidence_hit_rate"),
                }
            )
    return rows


def _summary_metric_lookup(run_summary: dict[str, Any] | None) -> dict[tuple[str, str], float]:
    if run_summary is None:
        return {}
    lookup: dict[tuple[str, str], float] = {}
    for row in run_summary.get("auxiliary_comparison_table", []):
        baseline = row.get("baseline")
        if not baseline:
            continue
        for metric, value in row.items():
            if metric == "baseline" or value is None:
                continue
            lookup[(baseline, metric)] = float(value)
    for row in run_summary.get("comparison_table", []):
        baseline = row.get("baseline")
        if not baseline:
            continue
        for metric, value in row.items():
            if metric == "baseline" or value is None:
                continue
            lookup[(baseline, metric)] = float(value)
    return lookup


def _apply_summary_overrides(
    summary_rows: list[dict[str, Any]],
    run_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    lookup = _summary_metric_lookup(run_summary)
    if not lookup:
        return summary_rows

    overridden: list[dict[str, Any]] = []
    for row in summary_rows:
        new_row = dict(row)
        new_row["mean_source"] = "bootstrap"
        if row.get("category") != "overall":
            overridden.append(new_row)
            continue
        official_mean = lookup.get((row["baseline"], row["metric"]))
        if official_mean is None:
            overridden.append(new_row)
            continue
        approx_mean = float(row["mean"])
        delta = official_mean - approx_mean
        new_row["mean"] = round(official_mean, 3)
        new_row["ci_low"] = round(float(row["ci_low"]) + delta, 3)
        new_row["ci_high"] = round(float(row["ci_high"]) + delta, 3)
        new_row["mean_source"] = "run_summary"
        overridden.append(new_row)
    return overridden


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(PROJECT_ROOT / args.db_path)
    answers_by_baseline = _load_answers_by_baseline(connection, args.run_id)
    artifacts_by_baseline = _load_artifacts(args.run_id)
    run_summary = _load_run_summary(args.run_id)
    primary_rows = _collect_primary_metric_values(args.run_id, answers_by_baseline, artifacts_by_baseline)["rows"]

    main_metrics = [
        "profile_selection_accuracy",
        "ontology_constrained_extraction_f1",
        "query_support_path_coverage",
        "answer_quality_delta",
    ]
    secondary_metrics = [
        "relation_schema_conformance_rate",
        "fallback_error_rate",
    ]

    main_summary = _apply_summary_overrides(
        _summarize_metric_rows(primary_rows, main_metrics, args.bootstrap_resamples, args.bootstrap_seed),
        run_summary,
    )
    supporting_summary = _apply_summary_overrides(
        _summarize_metric_rows(primary_rows, ["answer_quality_score"], args.bootstrap_resamples, args.bootstrap_seed),
        run_summary,
    )
    secondary_summary = _summarize_metric_rows(primary_rows, secondary_metrics, args.bootstrap_resamples, args.bootstrap_seed)

    manual_eval_path = _manual_eval_path(args)
    manual_eval = _load_manual_eval(manual_eval_path)
    manual_rows = _manual_rows(manual_eval)
    manual_summary = _summarize_metric_rows(
        manual_rows,
        ["required_answer_slot_coverage_manual", "preferred_answer_evidence_hit_rate"],
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )

    payload = {
        "run_id": args.run_id,
        "run_summary_path": str(_summary_path(args.run_id)) if _summary_path(args.run_id) else None,
        "manual_eval_path": str(manual_eval_path) if manual_eval_path else None,
        "main_metrics": main_summary,
        "supporting_metrics": supporting_summary,
        "secondary_metrics": secondary_summary + manual_summary,
        "notes": {
            "primary_sample_size_target": "50/category for main results; 90/category for robustness.",
            "manual_secondary_metrics_scope": "manual gold subset only when manual_eval_path is provided.",
            "answer_quality_delta_definition": "Per-example token-F1 difference against reference_only_baseline.",
            "bootstrap": {
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "confidence_interval": "95% percentile bootstrap CI",
            },
        },
    }

    json_path = output_dir / f"{args.run_id}_proposal_metrics_report.json"
    json_path.write_text(json.dumps(payload, indent=2))
    _write_csv(output_dir / f"{args.run_id}_proposal_main_metrics.csv", main_summary)
    _write_csv(output_dir / f"{args.run_id}_proposal_supporting_metrics.csv", supporting_summary)
    _write_csv(output_dir / f"{args.run_id}_proposal_secondary_metrics.csv", secondary_summary + manual_summary)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
