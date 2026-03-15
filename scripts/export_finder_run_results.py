from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FinDER run results into JSON and CSV.")
    parser.add_argument("--run-id", required=True, help="Experiment run id to export.")
    parser.add_argument(
        "--db-path",
        default="data/metadata/experiment.sqlite",
        help="Path to the experiment SQLite database.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/finder_runs",
        help="Directory to write exported files.",
    )
    return parser.parse_args()


def _load_run_record(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT run_id, run_type, agent_framework, model_name, dataset_path,
               ontology_version, started_at, finished_at, status, notes, created_at
        FROM experiment_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Run not found: {run_id}")
    keys = [
        "run_id",
        "run_type",
        "agent_framework",
        "model_name",
        "dataset_path",
        "ontology_version",
        "started_at",
        "finished_at",
        "status",
        "notes",
        "created_at",
    ]
    return dict(zip(keys, row, strict=True))


def _load_documents(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT doc_id, source_dataset, category, sample_type, input_text, reference_json, created_at
        FROM documents
        """
    ).fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[row[0]] = {
            "doc_id": row[0],
            "source_dataset": row[1],
            "category": row[2],
            "sample_type": row[3],
            "input_text": row[4],
            "reference_json": row[5],
            "created_at": row[6],
        }
    return result


def _load_question_answers(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT question_id, doc_id, category, query_template, selected_profile,
               answer_text, answer_confidence, supporting_edges_json, evaluation_json, created_at
        FROM question_answers
        WHERE run_id = ?
        ORDER BY question_id
        """,
        (run_id,),
    ).fetchall()
    answers: list[dict[str, Any]] = []
    for row in rows:
        evaluation = json.loads(row[8]) if row[8] else {}
        baseline = evaluation.get("baseline")
        if baseline is None and ":" in row[0]:
            baseline = row[0].split(":", 1)[0]
        answers.append(
            {
                "question_id": row[0],
                "doc_id": row[1],
                "category": row[2],
                "query_template": row[3],
                "selected_profile": row[4],
                "answer_text": row[5],
                "answer_confidence": row[6],
                "supporting_edges_json": row[7],
                "evaluation": evaluation,
                "baseline": baseline,
                "created_at": row[9],
            }
        )
    return answers


def _load_profile_decisions(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT doc_id, selected_profile, candidate_profiles_json, confidence,
               mapping_policy, extension_policy, rationale_json, created_at
        FROM profile_decisions
        WHERE run_id = ?
        ORDER BY doc_id
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "doc_id": row[0],
            "selected_profile": row[1],
            "candidate_profiles_json": row[2],
            "confidence": row[3],
            "mapping_policy": row[4],
            "extension_policy": row[5],
            "rationale_json": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]


def _load_artifacts(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT artifact_id, doc_id, question_id, artifact_type, artifact_path, metadata_json, created_at
        FROM artifacts
        WHERE run_id = ?
        ORDER BY artifact_type, doc_id
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "artifact_id": row[0],
            "doc_id": row[1],
            "question_id": row[2],
            "artifact_type": row[3],
            "artifact_path": row[4],
            "metadata_json": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]


def _load_graph_ingestion(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT doc_id, node_count, edge_count, graph_namespace, graph_uri, quality_summary_json, created_at
        FROM graph_ingestion
        WHERE run_id = ?
        ORDER BY doc_id
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "doc_id": row[0],
            "node_count": row[1],
            "edge_count": row[2],
            "graph_namespace": row[3],
            "graph_uri": row[4],
            "quality_summary_json": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]


def _load_summary_if_present(run_record: dict[str, Any]) -> dict[str, Any] | None:
    notes = run_record.get("notes") or ""
    candidate = Path(notes)
    if candidate.is_file():
        return json.loads(candidate.read_text())
    summary_path = ROOT_DIR / "data" / "experiment_runs" / f"{run_record['run_id']}_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    return None


def _load_checkpoint_if_present(run_record: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint_path = ROOT_DIR / "data" / "experiment_runs" / f"{run_record['run_id']}_checkpoint.json"
    if checkpoint_path.is_file():
        return json.loads(checkpoint_path.read_text())
    return None


def _load_artifact_payloads(run_id: str) -> list[dict[str, Any]]:
    artifact_root = ROOT_DIR / "data" / "experiment_runs" / "artifacts" / run_id
    payloads: list[dict[str, Any]] = []
    if not artifact_root.exists():
        return payloads
    for path in sorted(artifact_root.glob("*/*.json")):
        payload = json.loads(path.read_text())
        payload["_artifact_path"] = str(path)
        payloads.append(payload)
    return payloads


def _compute_partial_progress(
    answers: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    artifact_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    answers_by_baseline = Counter(answer["baseline"] for answer in answers if answer.get("baseline"))
    artifacts_by_baseline = Counter(artifact["artifact_type"] for artifact in artifacts)
    payloads_by_baseline = Counter(payload["baseline"] for payload in artifact_payloads)
    token_f1_by_baseline: dict[str, list[float]] = defaultdict(list)
    for answer in answers:
        baseline = answer.get("baseline")
        token_f1 = answer["evaluation"].get("token_f1")
        if baseline and token_f1 is not None:
            token_f1_by_baseline[baseline].append(float(token_f1))
    partial_answer_scores = {
        baseline: round(sum(values) / len(values), 3)
        for baseline, values in sorted(token_f1_by_baseline.items())
        if values
    }
    return {
        "question_answers_by_baseline": dict(sorted(answers_by_baseline.items())),
        "artifact_rows_by_baseline": dict(sorted(artifacts_by_baseline.items())),
        "artifact_payloads_by_baseline": dict(sorted(payloads_by_baseline.items())),
        "partial_answer_quality_score_by_baseline": partial_answer_scores,
    }


def _build_answer_rows(
    run_id: str,
    answers: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for answer in answers:
        document = documents.get(answer["doc_id"], {})
        rows.append(
            {
                "run_id": run_id,
                "baseline": answer.get("baseline"),
                "question_id": answer["question_id"],
                "doc_id": answer["doc_id"],
                "category": answer["category"] or document.get("category"),
                "sample_type": document.get("sample_type"),
                "question_text": document.get("input_text"),
                "selected_profile": answer["selected_profile"],
                "query_template": answer["query_template"],
                "intent_id": answer["evaluation"].get("intent_id"),
                "answer_confidence": answer["answer_confidence"],
                "token_f1": answer["evaluation"].get("token_f1"),
                "support_score": answer["evaluation"].get("support_score"),
                "answerable": answer["evaluation"].get("answerable"),
                "filled_slots_json": json.dumps(answer["evaluation"].get("filled_slots", [])),
                "missing_slots_json": json.dumps(answer["evaluation"].get("missing_slots", [])),
                "missing_requirements_json": json.dumps(answer["evaluation"].get("missing_requirements", [])),
                "answer_text": answer["answer_text"],
                "supporting_edges_json": answer["supporting_edges_json"],
                "created_at": answer["created_at"],
            }
        )
    return rows


def _build_artifact_rows(artifact_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in artifact_payloads:
        answer = payload.get("answer", {})
        profile_decision = payload.get("profile_decision", {})
        query_support = payload.get("query_support", {})
        extraction = payload.get("extraction", {})
        evidence_bundle = answer.get("evidence_bundle", {})
        rows.append(
            {
                "baseline": payload.get("baseline"),
                "example_id": payload.get("example_id"),
                "selected_profile": profile_decision.get("selected_profile"),
                "selection_confidence": profile_decision.get("selection_confidence"),
                "extraction_mode": extraction.get("metadata", {}).get("mode"),
                "evaluation_gold_source": extraction.get("evaluation_gold_source"),
                "intent_id": query_support.get("intent_id") or evidence_bundle.get("intent_id"),
                "entity_count": len(extraction.get("entities", [])),
                "relation_count": len(extraction.get("relations", [])),
                "support_score": query_support.get("support_score"),
                "answerable": query_support.get("answerable"),
                "filled_slots_json": json.dumps(query_support.get("filled_slots", [])),
                "missing_requirements_json": json.dumps(query_support.get("missing_requirements", [])),
                "token_f1": answer.get("token_f1"),
                "confidence": answer.get("confidence"),
                "artifact_path": payload.get("_artifact_path"),
            }
        )
    return rows


def _build_overview_rows(
    export_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = export_payload.get("summary") or {}
    partial = export_payload.get("partial_progress") or {}
    metrics_by_baseline = {
        row["baseline"]: row
        for row in summary.get("comparison_table", []) + summary.get("auxiliary_comparison_table", [])
    }
    baselines = sorted(
        set(partial.get("question_answers_by_baseline", {}).keys())
        | set(partial.get("artifact_payloads_by_baseline", {}).keys())
        | set(metrics_by_baseline.keys())
    )
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        metric_row = metrics_by_baseline.get(baseline, {})
        rows.append(
            {
                "run_id": export_payload["run"]["run_id"],
                "run_status": export_payload["run"]["status"],
                "summary_present": export_payload["summary_present"],
                "baseline": baseline,
                "question_answer_count": partial.get("question_answers_by_baseline", {}).get(baseline, 0),
                "artifact_payload_count": partial.get("artifact_payloads_by_baseline", {}).get(baseline, 0),
                "partial_answer_quality_score": partial.get("partial_answer_quality_score_by_baseline", {}).get(baseline),
                "profile_selection_accuracy": metric_row.get("profile_selection_accuracy"),
                "ontology_constrained_extraction_f1": metric_row.get("ontology_constrained_extraction_f1"),
                "query_support_path_coverage": metric_row.get("query_support_path_coverage"),
                "answer_quality_score": metric_row.get("answer_quality_score"),
                "answer_quality_delta": metric_row.get("answer_quality_delta"),
                "answer_quality_delta_vs_question_only": metric_row.get("answer_quality_delta_vs_question_only"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = parse_args()
    db_path = ROOT_DIR / args.db_path
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        run_record = _load_run_record(connection, args.run_id)
        documents = _load_documents(connection)
        answers = _load_question_answers(connection, args.run_id)
        profile_decisions = _load_profile_decisions(connection, args.run_id)
        artifacts = _load_artifacts(connection, args.run_id)
        graph_ingestion = _load_graph_ingestion(connection, args.run_id)

    summary = _load_summary_if_present(run_record)
    checkpoint = _load_checkpoint_if_present(run_record)
    artifact_payloads = _load_artifact_payloads(args.run_id)
    partial_progress = _compute_partial_progress(answers, artifacts, artifact_payloads)

    export_payload = {
        "run": run_record,
        "summary_present": summary is not None,
        "summary": summary,
        "checkpoint_present": checkpoint is not None,
        "checkpoint": checkpoint,
        "partial_progress": partial_progress,
        "counts": {
            "question_answers": len(answers),
            "profile_decisions": len(profile_decisions),
            "artifacts": len(artifacts),
            "graph_ingestion_rows": len(graph_ingestion),
            "artifact_payloads": len(artifact_payloads),
        },
        "profile_decisions": profile_decisions,
    }

    json_path = output_dir / f"{args.run_id}_export.json"
    answers_csv_path = output_dir / f"{args.run_id}_answers.csv"
    artifacts_csv_path = output_dir / f"{args.run_id}_artifacts.csv"
    overview_csv_path = output_dir / f"{args.run_id}_overview.csv"

    json_path.write_text(json.dumps(export_payload, indent=2))
    _write_csv(answers_csv_path, _build_answer_rows(args.run_id, answers, documents))
    _write_csv(artifacts_csv_path, _build_artifact_rows(artifact_payloads))
    _write_csv(overview_csv_path, _build_overview_rows(export_payload))

    print(json.dumps({
        "run_id": args.run_id,
        "json": str(json_path),
        "answers_csv": str(answers_csv_path),
        "artifacts_csv": str(artifacts_csv_path),
        "overview_csv": str(overview_csv_path),
        "summary_present": summary is not None,
        "partial_progress": partial_progress,
    }, indent=2))
