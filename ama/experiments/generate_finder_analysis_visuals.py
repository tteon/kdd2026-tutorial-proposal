from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path("docs/finder_category_signal_analysis.json")
OUTPUT_DIR = Path("docs/figures")
MARKDOWN_PATH = Path("docs/finder_analysis_visuals.md")


def load_analysis(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def short_category(name: str) -> str:
    return {
        "Company overview": "Company",
        "Shareholder return": "Shareholder",
    }.get(name, name)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }',
        ".title { font-size: 20px; font-weight: 700; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; }",
        ".axis { stroke: #9ca3af; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        '</style>',
    ]


def finish_svg(lines: list[str]) -> str:
    return "\n".join(lines + ["</svg>"])


def duplicate_rate_grouped_chart(data: dict[str, Any]) -> str:
    width, height = 1100, 520
    left, right, top, bottom = 90, 40, 70, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    categories = list(data["fields"]["references"].keys())
    series = [
        ("text", "#93c5fd"),
        ("references", "#2563eb"),
        ("answer", "#f59e0b"),
    ]

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="32" class="title">Duplicate Rate by Category and Field</text>')
    lines.append(f'<text x="{left}" y="52" class="small">Higher values indicate stronger lexical repetition.</text>')

    for step in range(6):
        value = step / 5
        y = top + chart_h - (value * chart_h)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid" />')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" class="label" text-anchor="end">{value:.1f}</text>')

    lines.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis" />')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis" />')

    group_w = chart_w / len(categories)
    bar_w = min(28, group_w / 4)
    offsets = [-bar_w, 0, bar_w]

    for index, category in enumerate(categories):
        x_center = left + (index + 0.5) * group_w
        lines.append(
            f'<text x="{x_center:.1f}" y="{height - 32}" class="label" text-anchor="middle">{short_category(category)}</text>'
        )
        for (field_name, color), offset in zip(series, offsets):
            rate = data["fields"][field_name][category]["duplicate_rate"]
            bar_h = rate * chart_h
            x = x_center + offset - (bar_w / 2)
            y = top + chart_h - bar_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3" />'
            )

    legend_x = width - 250
    for idx, (label, color) in enumerate(series):
        y = 28 + (idx * 20)
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{color}" rx="2" />')
        lines.append(f'<text x="{legend_x + 18}" y="{y}" class="label">{label}</text>')

    return finish_svg(lines)


def cleaned_reference_bar_chart(data: dict[str, Any]) -> str:
    width, height = 920, 420
    left, right, top, bottom = 170, 40, 70, 40
    chart_w = width - left - right
    row_h = 34
    categories = sorted(
        data["cleaned_references"].items(),
        key=lambda item: (-item[1]["duplicate_rate"], item[0]),
    )

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="32" class="title">Cleaned Reference Duplicate Rate Ranking</text>')
    lines.append(f'<text x="{left}" y="52" class="small">Boilerplate-like terms removed before scoring.</text>')
    lines.append(f'<line x1="{left}" y1="{top + len(categories)*row_h}" x2="{left + chart_w}" y2="{top + len(categories)*row_h}" class="axis" />')

    for step in range(6):
        value = step / 5
        x = left + value * chart_w
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + len(categories)*row_h}" class="grid" />')
        lines.append(f'<text x="{x:.1f}" y="{top + len(categories)*row_h + 18}" class="label" text-anchor="middle">{value:.1f}</text>')

    for idx, (category, stats) in enumerate(categories):
        y = top + idx * row_h
        rate = stats["duplicate_rate"]
        bar_w = rate * chart_w
        lines.append(f'<text x="{left - 10}" y="{y + 20}" class="label" text-anchor="end">{category}</text>')
        lines.append(f'<rect x="{left}" y="{y + 6}" width="{bar_w:.1f}" height="20" fill="#10b981" rx="3" />')
        lines.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 20}" class="label">{rate:.4f}</text>')

    return finish_svg(lines)


def shared_term_heatmap(data: dict[str, Any]) -> str:
    width, height = 980, 420
    left, right, top, bottom = 170, 30, 90, 70
    chart_w = width - left - right
    chart_h = height - top - bottom

    selected_terms = ["market", "price", "loss", "value", "assets"]
    term_lookup = {item["term"]: item for item in data["shared_reference_term_contexts"]}
    categories = list(data["fields"]["references"].keys())
    cell_w = chart_w / len(categories)
    cell_h = chart_h / len(selected_terms)

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="32" class="title">Shared-Term Context Heatmap</text>')
    lines.append(f'<text x="{left}" y="52" class="small">Cell value = document rate for the shared term within that category.</text>')

    for col, category in enumerate(categories):
        x = left + col * cell_w + (cell_w / 2)
        lines.append(f'<text x="{x:.1f}" y="{top - 12}" class="label" text-anchor="middle">{short_category(category)}</text>')

    for row, term in enumerate(selected_terms):
        y = top + row * cell_h
        lines.append(f'<text x="{left - 10}" y="{y + cell_h/2 + 4:.1f}" class="label" text-anchor="end">{term}</text>')
        contexts = {ctx["category"]: ctx["document_rate"] for ctx in term_lookup.get(term, {}).get("contexts", [])}
        for col, category in enumerate(categories):
            x = left + col * cell_w
            rate = contexts.get(category, 0.0)
            intensity = int(245 - (rate * 170))
            fill = f"rgb({intensity},{intensity},{255})"
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" />')
            lines.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 4:.1f}" class="small" text-anchor="middle">{rate:.2f}</text>')

    return finish_svg(lines)


def phrase_signal_heatmap(data: dict[str, Any]) -> str:
    width, height = 980, 430
    left, right, top, bottom = 230, 30, 90, 50
    chart_w = width - left - right
    chart_h = height - top - bottom

    categories = list(data["fields"]["references"].keys())
    phrases = []
    for category in categories:
        top_phrase = data["category_specific_reference_phrases"][category][0]["phrase"]
        phrases.append((category, top_phrase))

    cell_w = chart_w / len(categories)
    cell_h = chart_h / len(phrases)

    # Build phrase-by-category matrix using category-specific phrase rates where available
    phrase_rates: dict[str, dict[str, float]] = {}
    for category in categories:
        phrase_rates[category] = {
            item["phrase"]: item["category_document_rate"]
            for item in data["category_specific_reference_phrases"][category]
        }

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="32" class="title">Category-Specific Phrase Signal Heatmap</text>')
    lines.append(f'<text x="{left}" y="52" class="small">Diagonal cells show each category’s strongest phrase-level signal.</text>')

    for col, category in enumerate(categories):
        x = left + col * cell_w + (cell_w / 2)
        lines.append(f'<text x="{x:.1f}" y="{top - 12}" class="label" text-anchor="middle">{short_category(category)}</text>')

    for row, (owner_category, phrase) in enumerate(phrases):
        y = top + row * cell_h
        lines.append(f'<text x="{left - 12}" y="{y + cell_h/2 + 4:.1f}" class="label" text-anchor="end">{phrase}</text>')
        for col, category in enumerate(categories):
            x = left + col * cell_w
            rate = phrase_rates.get(category, {}).get(phrase, 0.0)
            intensity = int(245 - (rate * 180))
            fill = f"rgb(255,{intensity},{intensity})"
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" />')
            text_fill = "#111827" if rate < 0.45 else "#ffffff"
            lines.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 4:.1f}" class="small" text-anchor="middle" fill="{text_fill}">{rate:.2f}</text>')

    return finish_svg(lines)


def top_enriched_tables(data: dict[str, Any]) -> tuple[str, str, str]:
    duplicate_lines = [
        "| Category | Text dup | Ref dup | Cleaned ref dup | Answer dup |",
        "|---|---:|---:|---:|---:|",
    ]
    categories = list(data["fields"]["references"].keys())
    for category in categories:
        duplicate_lines.append(
            "| {category} | {text:.4f} | {ref:.4f} | {clean:.4f} | {answer:.4f} |".format(
                category=category,
                text=data["fields"]["text"][category]["duplicate_rate"],
                ref=data["fields"]["references"][category]["duplicate_rate"],
                clean=data["cleaned_references"][category]["duplicate_rate"],
                answer=data["fields"]["answer"][category]["duplicate_rate"],
            )
        )

    term_lines = [
        "| Category | Top enriched term | Category rate | Outside rate | χ² |",
        "|---|---|---:|---:|---:|",
    ]
    for category in categories:
        item = data["category_specific_reference_terms"][category][0]
        term_lines.append(
            f"| {category} | `{item['term']}` | {item['category_document_rate']:.4f} | {item['outside_document_rate']:.4f} | {item['chi_square']:.2f} |"
        )

    phrase_lines = [
        "| Category | Top enriched phrase | Category rate | Outside rate | χ² |",
        "|---|---|---:|---:|---:|",
    ]
    for category in categories:
        item = data["category_specific_reference_phrases"][category][0]
        phrase_lines.append(
            f"| {category} | `{item['phrase']}` | {item['category_document_rate']:.4f} | {item['outside_document_rate']:.4f} | {item['chi_square']:.2f} |"
        )

    return "\n".join(duplicate_lines), "\n".join(term_lines), "\n".join(phrase_lines)


def build_markdown(data: dict[str, Any]) -> str:
    duplicate_table, term_table, phrase_table = top_enriched_tables(data)
    reasoning_lines = [
        "| Category | Reasoning=true rate | Avg ref tokens (true) | Avg ref tokens (false) |",
        "|---|---:|---:|---:|",
    ]
    for category, stats in data["reasoning_flags"].items():
        reasoning_lines.append(
            f"| {category} | {stats['reasoning_true_rate']:.4f} | {stats['avg_reference_tokens_when_reasoning_true']:.2f} | {stats['avg_reference_tokens_when_reasoning_false']:.2f} |"
        )
    reasoning_table = "\n".join(reasoning_lines)

    return f"""# FinDER Analysis Visuals

This document turns the existing lexical and phrase analysis into presentation-friendly tables and SVG figures.

## Figure 1 — Duplicate rate by field

![Duplicate rate by category and field](figures/finder_duplicate_rate_grouped.svg)

## Figure 2 — Cleaned reference duplicate rate ranking

![Cleaned reference duplicate rate ranking](figures/finder_cleaned_reference_duplicate_rates.svg)

## Figure 3 — Shared-term heatmap

![Shared-term heatmap](figures/finder_shared_term_heatmap.svg)

## Figure 4 — Phrase signal heatmap

![Phrase signal heatmap](figures/finder_phrase_signal_heatmap.svg)

## Table 1 — Duplicate rate summary

{duplicate_table}

## Table 2 — Strongest enriched term per category

{term_table}

## Table 3 — Strongest enriched phrase per category

{phrase_table}

## Table 4 — Reasoning slice summary

{reasoning_table}

## Notes

- Figure 1 highlights how much stronger `references` is than `text` as a category signal source.
- Figure 2 shows that the ranking largely survives after removing common reporting vocabulary.
- Figure 3 shows that shared terms still distribute differently across categories, supporting the context-preservation argument.
- Figure 4 shows how sharply the strongest phrase proxies concentrate by category.
- Tables 2 and 3 give a compact “entity-like” view of the strongest category markers.
"""


def main() -> None:
    data = load_analysis(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_text(OUTPUT_DIR / "finder_duplicate_rate_grouped.svg", duplicate_rate_grouped_chart(data))
    write_text(OUTPUT_DIR / "finder_cleaned_reference_duplicate_rates.svg", cleaned_reference_bar_chart(data))
    write_text(OUTPUT_DIR / "finder_shared_term_heatmap.svg", shared_term_heatmap(data))
    write_text(OUTPUT_DIR / "finder_phrase_signal_heatmap.svg", phrase_signal_heatmap(data))
    write_text(MARKDOWN_PATH, build_markdown(data))
    print(f"Wrote visuals to {OUTPUT_DIR}")
    print(f"Wrote summary to {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
