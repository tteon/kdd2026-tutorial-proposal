#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROPOSAL_DIR = ROOT_DIR / "proposal"
FIGURE_DIR = PROPOSAL_DIR / "figures"
RUN_DIR = ROOT_DIR / "data" / "experiment_runs"
REPORT_DIR = ROOT_DIR / "exports" / "proposal_metrics"

GRAPH_BASELINES = [
    "graph_without_ontology_constraints",
    "graph_with_profile_selection_only",
    "graph_with_profile_plus_constrained_extraction_and_linking",
    "full_minimal_pipeline_with_quality_aware_evidence_selection",
]

BASELINE_LABELS = {
    "graph_without_ontology_constraints": "Graph",
    "graph_with_profile_selection_only": "Graph+Profile",
    "graph_with_profile_plus_constrained_extraction_and_linking": "Graph+Constraint",
    "full_minimal_pipeline_with_quality_aware_evidence_selection": "Full",
}

BASELINE_COLORS = {
    "graph_without_ontology_constraints": "#d95f02",
    "graph_with_profile_selection_only": "#7570b3",
    "graph_with_profile_plus_constrained_extraction_and_linking": "#1b9e77",
    "full_minimal_pipeline_with_quality_aware_evidence_selection": "#2a9d8f",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _report_lookup(report: dict) -> dict[tuple[str, str, str], dict]:
    lookup: dict[tuple[str, str, str], dict] = {}
    for section in ("main_metrics", "secondary_metrics", "supporting_metrics"):
        for item in report.get(section, []):
            lookup[(item["baseline"], item["category"], item["metric"])] = item
    return lookup


def _get_metric(lookup: dict, baseline: str, metric: str, category: str = "overall") -> dict | None:
    return lookup.get((baseline, category, metric))


def _fmt_ci(item: dict | None) -> str:
    if not item:
        return "-"
    mean = item.get("mean")
    low = item.get("ci_low")
    high = item.get("ci_high")
    if mean is None:
        return "-"
    if low is None or high is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _build_figure_svg() -> str:
    width = 1360
    height = 720
    dark = "#102a43"
    muted = "#52606d"
    edge = "#cbd2d9"
    accent = "#1f6f8b"
    graph_fill = "#eef7f3"
    aux_fill = "#eef4f8"
    data_fill = "#f8f5ef"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="40" y="46" font-family="Helvetica, Arial, sans-serif" font-size="28" fill="#102a43">Figure 1. Minimal Ontology-Guided Finance Graph Pipeline</text>',
        '<text x="40" y="74" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#52606d">One shared answer agent is used across baselines; only the evidence source changes.</text>',
    ]

    top_boxes = [
        ("FinDER\nreferences", 50, 140, 160, 78, data_fill),
        ("Profile\nselection", 250, 140, 150, 78, "#ffffff"),
        ("Ontology-constrained\nextraction", 440, 140, 200, 78, "#ffffff"),
        ("Entity linking\nand graph build", 680, 140, 180, 78, "#ffffff"),
        ("Intent-aware query\nsupport analysis", 900, 140, 210, 78, "#ffffff"),
        ("Evidence bundle\nentities | triples |\nsnippets | missing slots", 1150, 132, 170, 94, graph_fill),
    ]
    for label, x, y, w, h, fill in top_boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{edge}" stroke-width="2"/>')
        for idx, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{x + w/2}" y="{y + 28 + idx*18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="{dark}">{line}</text>')

    for x1, x2 in [(210, 250), (400, 440), (640, 680), (860, 900), (1110, 1150)]:
        y = 179
        parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{accent}" stroke-width="4"/>')
        parts.append(f'<polygon points="{x2},{y} {x2-12},{y-6} {x2-12},{y+6}" fill="{accent}"/>')

    parts.append(f'<text x="52" y="285" font-family="Helvetica, Arial, sans-serif" font-size="18" fill="{dark}">Baseline comparison under a shared answer agent</text>')

    lower_boxes = [
        ("Question only", 90, 330, 170, 68, data_fill),
        ("Reference text only", 320, 330, 190, 68, aux_fill),
        ("Graph evidence bundle", 590, 330, 210, 68, graph_fill),
        ("Shared answer\nagent", 900, 318, 170, 92, "#ffffff"),
        ("FinDER answer\nevaluation", 1150, 318, 170, 92, "#ffffff"),
    ]
    for label, x, y, w, h, fill in lower_boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{edge}" stroke-width="2"/>')
        for idx, line in enumerate(label.split("\n")):
            parts.append(f'<text x="{x + w/2}" y="{y + 28 + idx*18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" fill="{dark}">{line}</text>')

    for (x1, x2) in [(260, 320), (510, 590), (800, 900), (1070, 1150)]:
        y = 364
        parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{accent}" stroke-width="4"/>')
        parts.append(f'<polygon points="{x2},{y} {x2-12},{y-6} {x2-12},{y+6}" fill="{accent}"/>')

    parts.append(f'<line x1="1235" y1="226" x2="1235" y2="318" stroke="#2a9d8f" stroke-width="4"/>')
    parts.append(f'<polygon points="1235,318 1229,306 1241,306" fill="#2a9d8f"/>')

    notes = [
        ("Primary graph ablation: Graph -> Graph+Profile -> Graph+Constraint -> Full", 60, 550),
        ("Auxiliary text baselines are retained for interpretation but not emphasized in the main graph comparison.", 60, 580),
        ("Evaluation compares generated answers against FinDER ground-truth answers.", 60, 610),
    ]
    for text, x, y in notes:
        parts.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="{muted}">{text}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _build_chart_svg(summary: dict, report_lookup: dict) -> str:
    width = 1280
    height = 700
    dark = "#102a43"
    muted = "#52606d"
    bg = "#fbfaf7"
    panel_fill = "#ffffff"
    left = 110
    chart_w = 1040
    panel_y = 140
    panel_h = 400

    answer_scores: dict[str, float] = {}
    coverage_scores: dict[str, float] = {}
    for row in summary.get("comparison_table", []):
        answer_scores[row["baseline"]] = float(row.get("answer_quality_score") or 0.0)
        coverage_scores[row["baseline"]] = float(row.get("query_support_path_coverage") or 0.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
        '<text x="40" y="46" font-family="Helvetica, Arial, sans-serif" font-size="28" fill="#102a43">Chart 1. Core Evidence for the Tutorial Hypothesis</text>',
        '<text x="40" y="74" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#52606d">Bars show answer quality; the line shows query-support path coverage.</text>',
        f'<rect x="40" y="{panel_y-36}" width="{width-80}" height="{panel_h+120}" rx="18" fill="{panel_fill}" stroke="#e4e7eb"/>',
    ]

    max_answer = 0.36
    max_cov = 0.70
    bar_w = 155
    gap = 80
    start_x = 140

    for tick in range(7):
        value = max_answer * tick / 6
        y = panel_y + panel_h - (panel_h * value / max_answer)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left-52}" y="{y+4}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{muted}">{value:.2f}</text>')
    parts.append(f'<text x="{left-92}" y="{panel_y-10}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{muted}">Answer quality</text>')

    for tick in range(7):
        value = max_cov * tick / 6
        y = panel_y + panel_h - (panel_h * value / max_cov)
        parts.append(f'<text x="{left + chart_w + 16}" y="{y+4}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{muted}">{value:.2f}</text>')
    parts.append(f'<text x="{left + chart_w - 8}" y="{panel_y-10}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{muted}">Coverage</text>')

    points = []
    for idx, baseline in enumerate(GRAPH_BASELINES):
        score = answer_scores.get(baseline, 0.0)
        x = start_x + idx * (bar_w + gap)
        h = panel_h * score / max_answer
        y = panel_y + panel_h - h
        color = BASELINE_COLORS[baseline]
        label = BASELINE_LABELS[baseline]
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" rx="8" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2}" y="{panel_y + panel_h + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{dark}">{label}</text>')
        parts.append(f'<text x="{x + bar_w/2}" y="{y - 8}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{dark}">{score:.3f}</text>')
        cov = coverage_scores.get(baseline, 0.0)
        cx = x + bar_w / 2
        cy = panel_y + panel_h - (panel_h * cov / max_cov)
        points.append((cx, cy, cov))

    if points:
        parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y, _ in points)}" fill="none" stroke="#111827" stroke-width="4"/>')
        for cx, cy, cov in points:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="#111827"/>')
            parts.append(f'<text x="{cx}" y="{cy - 12}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111827">{cov:.3f}</text>')

    parts.append(f'<text x="60" y="610" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="{muted}">Reading: profile selection alone does not move answer quality. The large gain appears when ontology-constrained extraction and linking create query-supporting graphs.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _build_table_rows(report_lookup: dict) -> list[dict]:
    rows = []
    graph_anchor = _get_metric(report_lookup, "graph_without_ontology_constraints", "answer_quality_score")
    graph_anchor_mean = graph_anchor["mean"] if graph_anchor else 0.0
    for baseline in GRAPH_BASELINES:
        item = _get_metric(report_lookup, baseline, "answer_quality_score")
        graph_delta = None
        if item:
            graph_delta = {"mean": item["mean"] - graph_anchor_mean}
        rows.append(
            {
                "baseline": baseline,
                "label": BASELINE_LABELS[baseline],
                "answer_quality_score": _fmt_ci(_get_metric(report_lookup, baseline, "answer_quality_score")),
                "answer_quality_delta": _fmt_ci(graph_delta),
                "profile_selection_accuracy": _fmt_ci(_get_metric(report_lookup, baseline, "profile_selection_accuracy")),
                "ontology_constrained_extraction_f1": _fmt_ci(_get_metric(report_lookup, baseline, "ontology_constrained_extraction_f1")),
                "query_support_path_coverage": _fmt_ci(_get_metric(report_lookup, baseline, "query_support_path_coverage")),
                "relation_schema_conformance_rate": _fmt_ci(_get_metric(report_lookup, baseline, "relation_schema_conformance_rate")),
                "fallback_error_rate": _fmt_ci(_get_metric(report_lookup, baseline, "fallback_error_rate")),
            }
        )
    return rows


def _write_table(rows: list[dict], md_path: Path, csv_path: Path) -> None:
    columns = [
        "label",
        "answer_quality_score",
        "answer_quality_delta",
        "profile_selection_accuracy",
        "ontology_constrained_extraction_f1",
        "query_support_path_coverage",
        "relation_schema_conformance_rate",
        "fallback_error_rate",
    ]
    labels = {
        "label": "Baseline",
        "answer_quality_score": "Answer Quality",
        "answer_quality_delta": "Delta vs Graph",
        "profile_selection_accuracy": "Profile Acc",
        "ontology_constrained_extraction_f1": "Extraction F1",
        "query_support_path_coverage": "Coverage",
        "relation_schema_conformance_rate": "Schema Conformance",
        "fallback_error_rate": "Fallback Error Rate",
    }
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in columns})

    md_lines = [
        "# Table 1. Graph-Centered Shared Answer-Agent Results",
        "",
        "| " + " | ".join(labels[col] for col in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    md_lines.extend(
        [
            "",
            "Notes:",
            "",
            "- This main table is restricted to graph baselines to keep the narrative centered on ontology-guided ablations.",
            "- Values are overall means on the `50/category` shared answer-agent run.",
            "- Brackets denote 95% bootstrap CI when available.",
            "- `Delta vs Graph` uses `graph_without_ontology_constraints` as the anchor baseline.",
        ]
    )
    md_path.write_text("\n".join(md_lines))


def _write_latex_assets(output_dir: Path) -> None:
    figure_tex = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure1_pipeline.pdf}
\caption{Minimal ontology-guided finance graph pipeline. FinDER references are converted into profile-conditioned graphs, evaluated for intent-aware query support, and compared under a shared answer agent so that only the evidence source changes across baselines.}
\label{fig:finance-pipeline}
\end{figure*}
"""
    chart_tex = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/chart1_hypothesis_evidence.pdf}
\caption{Core evidence for the tutorial hypothesis on the shared answer-agent primary run. Answer quality improves only when ontology-constrained graph construction creates query-supporting graphs. Bars show answer quality and the line shows query-support path coverage.}
\label{fig:hypothesis-chart}
\end{figure}
"""
    table_tex = r"""\begin{table*}[t]
\centering
\caption{Graph-centered shared answer-agent results on the FinDER primary run (50 examples per category). Auxiliary text baselines are omitted here to keep the main table focused on ontology-guided graph ablations.}
\label{tab:primary-graph-results}
\begin{tabular}{lcccccc}
\toprule
Baseline & Answer Quality & $\Delta$ vs Graph & Profile Acc. & Extraction F1 & Coverage & Schema Conf. \\
\midrule
Graph & 0.175 [0.165, 0.184] & 0.000 & -- & 0.126 [0.103, 0.151] & 0.000 [0.000, 0.000] & 0.000 [0.000, 0.000] \\
Graph+Profile & 0.175 [0.166, 0.184] & 0.000 & 0.900 [0.847, 0.940] & 0.104 [0.082, 0.130] & 0.000 [0.000, 0.000] & 0.000 [0.000, 0.000] \\
Graph+Constraint & 0.318 [0.297, 0.338] & 0.143 & 0.900 [0.847, 0.940] & 0.083 [0.065, 0.104] & 0.589 [0.537, 0.638] & 1.000 [1.000, 1.000] \\
Full & 0.321 [0.300, 0.341] & 0.146 & 0.900 [0.847, 0.940] & 0.083 [0.065, 0.104] & 0.589 [0.537, 0.638] & 1.000 [1.000, 1.000] \\
\bottomrule
\end{tabular}
\end{table*}
"""
    (output_dir / "figure1_pipeline.tex").write_text(figure_tex)
    (output_dir / "chart1_hypothesis_evidence.tex").write_text(chart_tex)
    (output_dir / "table1_primary_results.tex").write_text(table_tex)


def _write_readme(output_dir: Path) -> None:
    readme = """# Proposal Figures

This directory contains the submission-facing assets for the current main run.

- `figure1_pipeline.svg`
  Main pipeline figure.
- `figure1_pipeline.tex`
  LaTeX wrapper for Figure 1.
- `chart1_hypothesis_evidence.svg`
  Main hypothesis chart.
- `chart1_hypothesis_evidence.tex`
  LaTeX wrapper for Chart 1.
- `table1_primary_results.md`
  Main graph-centered numeric table.
- `table1_primary_results.csv`
  CSV version of Table 1.
- `table1_primary_results.tex`
  LaTeX-ready academic table.

Suggested captions:

- Figure 1. Minimal ontology-guided finance graph pipeline. FinDER references are converted into profile-conditioned graphs, evaluated for intent-aware query support, and compared under a shared answer agent so that only the evidence source changes across baselines.
- Chart 1. Core evidence for the tutorial hypothesis on the shared answer-agent primary run. Answer quality improves only when ontology-constrained graph construction creates query-supporting graphs.
- Table 1. Graph-centered shared answer-agent results on FinDER `50/category`. Most gains appear only after constrained extraction and linking.
"""
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proposal-facing figure, chart, and table assets.")
    parser.add_argument("--run-id", default="finder-20260315013504-2f8ea4d3")
    args = parser.parse_args()

    summary_path = RUN_DIR / f"{args.run_id}_summary.json"
    report_path = REPORT_DIR / f"{args.run_id}_proposal_metrics_report.json"
    summary = _load_json(summary_path)
    report = _load_json(report_path)
    lookup = _report_lookup(report)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    (FIGURE_DIR / "figure1_pipeline.svg").write_text(_build_figure_svg())
    (FIGURE_DIR / "chart1_hypothesis_evidence.svg").write_text(_build_chart_svg(summary, lookup))
    rows = _build_table_rows(lookup)
    _write_table(rows, FIGURE_DIR / "table1_primary_results.md", FIGURE_DIR / "table1_primary_results.csv")
    _write_latex_assets(FIGURE_DIR)
    _write_readme(FIGURE_DIR)

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "figure1": str(FIGURE_DIR / "figure1_pipeline.svg"),
                "figure1_tex": str(FIGURE_DIR / "figure1_pipeline.tex"),
                "chart1": str(FIGURE_DIR / "chart1_hypothesis_evidence.svg"),
                "chart1_tex": str(FIGURE_DIR / "chart1_hypothesis_evidence.tex"),
                "table1_markdown": str(FIGURE_DIR / "table1_primary_results.md"),
                "table1_csv": str(FIGURE_DIR / "table1_primary_results.csv"),
                "table1_tex": str(FIGURE_DIR / "table1_primary_results.tex"),
                "readme": str(FIGURE_DIR / "README.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
