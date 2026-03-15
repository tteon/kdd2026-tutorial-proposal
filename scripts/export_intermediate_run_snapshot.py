#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RUN_DIR = DATA_DIR / "experiment_runs"
DB_PATH = DATA_DIR / "metadata" / "experiment.sqlite"
OUTPUT_DIR = ROOT_DIR / "exports" / "readable_results"


def _safe_slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace("_", "-")
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_checkpoint(run_id: str) -> dict:
    checkpoint_path = RUN_DIR / f"{run_id}_checkpoint.json"
    summary_path = RUN_DIR / f"{run_id}_summary.json"
    if checkpoint_path.exists():
        payload = _load_json(checkpoint_path)
        payload["_source_path"] = str(checkpoint_path)
        payload["_source_kind"] = "checkpoint"
        return payload
    if summary_path.exists():
        payload = _load_json(summary_path)
        payload["_source_path"] = str(summary_path)
        payload["_source_kind"] = "summary"
        return payload
    raise SystemExit(f"No checkpoint or summary found for {run_id}")


def _fetch_answer_rows(run_id: str) -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    overall_rows = conn.execute(
        """
        select
            substr(question_id, 1, instr(question_id, ':') - 1) as baseline,
            count(*) as example_count,
            round(avg(json_extract(evaluation_json, '$.token_f1')), 3) as answer_quality_score,
            round(avg(answer_confidence), 3) as avg_answer_confidence,
            round(avg(json_extract(evaluation_json, '$.support_score')), 3) as avg_support_score,
            round(avg(json_extract(evaluation_json, '$.answerable')), 3) as answerable_rate
        from question_answers
        where run_id = ?
        group by baseline
        order by baseline
        """,
        (run_id,),
    ).fetchall()
    category_rows = conn.execute(
        """
        select
            substr(question_id, 1, instr(question_id, ':') - 1) as baseline,
            category,
            count(*) as example_count,
            round(avg(json_extract(evaluation_json, '$.token_f1')), 3) as answer_quality_score,
            round(avg(answer_confidence), 3) as avg_answer_confidence,
            round(avg(json_extract(evaluation_json, '$.support_score')), 3) as avg_support_score,
            round(avg(json_extract(evaluation_json, '$.answerable')), 3) as answerable_rate
        from question_answers
        where run_id = ?
        group by baseline, category
        order by baseline, category
        """,
        (run_id,),
    ).fetchall()
    return [dict(r) for r in overall_rows], [dict(r) for r in category_rows]


def _build_svg(run_id: str, rows: list[dict]) -> str:
    labels = [row["baseline"] for row in rows]
    values = [float(row["answer_quality_score"] or 0.0) for row in rows]
    max_val = max(values) if values else 1.0
    width = 860
    height = 360
    margin_left = 230
    chart_width = 560
    row_h = 56
    top = 50
    bar_h = 28
    colors = {
        "question_only_baseline": "#9d7c4d",
        "reference_only_baseline": "#1f6f8b",
        "graph_without_ontology_constraints": "#d95f02",
        "graph_with_profile_selection_only": "#7570b3",
        "graph_with_profile_plus_constrained_extraction_and_linking": "#1b9e77",
        "full_minimal_pipeline_with_quality_aware_evidence_selection": "#2a9d8f",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f4ef"/>',
        f'<text x="30" y="32" font-family="Helvetica, Arial, sans-serif" font-size="22" fill="#1f2933">Intermediate FinDER Run Snapshot</text>',
        f'<text x="30" y="52" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#52606d">{run_id}</text>',
    ]
    for tick in range(6):
        x = margin_left + chart_width * tick / 5
        parts.append(f'<line x1="{x}" y1="70" x2="{x}" y2="{height-35}" stroke="#d9e2ec" stroke-width="1"/>')
        label = f"{(max_val * tick / 5):.2f}"
        parts.append(f'<text x="{x-10}" y="{height-15}" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#7b8794">{label}</text>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * row_h
        bar_w = 0 if max_val <= 0 else chart_width * value / max_val
        color = colors.get(label, "#486581")
        parts.append(f'<text x="30" y="{y + 20}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#102a43">{label}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{chart_width}" height="{bar_h}" rx="4" fill="#e4e7eb"/>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{margin_left + min(bar_w + 10, chart_width - 12)}" y="{y + 19}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#102a43">{value:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.3f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an intermediate readable run snapshot.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.run_id)
    overall_rows, category_rows = _fetch_answer_rows(args.run_id)

    baseline_progress = checkpoint.get("baseline_progress", {})
    overall_payload = []
    for row in overall_rows:
        progress = baseline_progress.get(row["baseline"], {}).get("progress", {})
        metrics = baseline_progress.get(row["baseline"], {}).get("metrics", {})
        merged = dict(row)
        merged["status"] = progress.get("status")
        merged["processed_examples"] = progress.get("processed_examples", row["example_count"])
        merged["total_examples"] = progress.get("total_examples")
        for key, value in metrics.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
        overall_payload.append(merged)

    category_map: dict[str, list[dict]] = defaultdict(list)
    for row in category_rows:
        category_map[row["baseline"]].append(row)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    status_slug = _safe_slug(checkpoint.get("status") or "unknown")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_intermediate-shared-answer-agent_50-per-category_{status_slug}_snapshot__run-{args.run_id}"

    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    svg_path = OUTPUT_DIR / f"{stem}.svg"

    snapshot = {
        "run_id": args.run_id,
        "source_kind": checkpoint.get("_source_kind"),
        "source_path": checkpoint.get("_source_path"),
        "status": checkpoint.get("status"),
        "stage": checkpoint.get("stage"),
        "current_baseline": checkpoint.get("current_baseline"),
        "sample_count": checkpoint.get("sample_count"),
        "updated_at": checkpoint.get("updated_at"),
        "baselines": overall_payload,
        "category_breakdown": category_rows,
    }
    json_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))

    md_lines = [
        "# Intermediate Run Snapshot",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Source: `{checkpoint.get('_source_kind')}`",
        f"- Status: `{checkpoint.get('status')}`",
        f"- Stage: `{checkpoint.get('stage')}`",
        f"- Current baseline: `{checkpoint.get('current_baseline')}`",
        f"- Sample count: `{checkpoint.get('sample_count')}`",
        f"- Updated at: `{checkpoint.get('updated_at')}`",
        "",
        "## Overall",
        "",
        _markdown_table(
            overall_payload,
            [
                "baseline",
                "status",
                "processed_examples",
                "total_examples",
                "answer_quality_score",
                "avg_answer_confidence",
                "ontology_constrained_extraction_f1",
                "query_support_path_coverage",
                "profile_selection_accuracy",
                "avg_support_score",
                "answerable_rate",
            ],
        ),
        "",
        "## Category Breakdown",
        "",
    ]
    for baseline, rows in category_map.items():
        md_lines.append(f"### {baseline}")
        md_lines.append("")
        md_lines.append(
            _markdown_table(
                rows,
                [
                    "category",
                    "example_count",
                    "answer_quality_score",
                    "avg_answer_confidence",
                    "avg_support_score",
                    "answerable_rate",
                ],
            )
        )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines))

    svg_path.write_text(_build_svg(args.run_id, overall_payload))

    print(json.dumps(
        {
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "svg_path": str(svg_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
