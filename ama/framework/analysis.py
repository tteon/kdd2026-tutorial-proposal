"""
Analysis utilities for inspecting experiment outputs.

  category_duplicate_entity_summary — duplicate entity stats per category
  rdf_lpg_comparison_summary        — side-by-side RDF vs LPG metrics
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def category_duplicate_entity_summary(
    artifacts: list[dict[str, Any]],
    *,
    category_key: str = "category",
    entities_key: str = "linked_entities",
    entity_name_key: str = "canonical_name",
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)

    for artifact in artifacts:
        category = str(artifact.get(category_key, "unknown"))
        for entity in artifact.get(entities_key, []):
            name = str(entity.get(entity_name_key, "")).strip().lower()
            if name:
                grouped[category].append(name)

    for category, entities in grouped.items():
        counts = Counter(entities)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        summary[category] = {
            "total_linked_entities": len(entities),
            "unique_linked_entities": len(counts),
            "duplicate_mentions": sum(count - 1 for count in duplicates.values()),
            "duplicate_entity_count": len(duplicates),
            "top_duplicates": dict(sorted(duplicates.items(), key=lambda item: (-item[1], item[0]))[:10]),
        }

    return summary


def _metrics_for_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate self-consistency metrics across a list of artifact dicts."""
    completed = [a for a in artifacts if a.get("status") == "completed"]
    n = len(completed)
    if n == 0:
        return {"count": 0}

    def avg(key: str) -> float | None:
        values = [a["metrics"][key] for a in completed if a.get("metrics", {}).get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    entity_counts = [len(a.get("extracted_entities", [])) for a in completed]
    linked_counts = [len(a.get("linked_entities", [])) for a in completed]

    return {
        "count": n,
        "avg_entity_count": round(sum(entity_counts) / n, 1),
        "avg_linked_count": round(sum(linked_counts) / n, 1),
        "avg_null_link_rate": avg("null_link_rate"),
        "avg_ambiguous_link_rate": avg("ambiguous_link_rate"),
        "avg_duplicate_entity_rate": avg("duplicate_entity_rate"),
        "avg_materialization_success_rate": avg("materialization_success_rate"),
    }


def rdf_lpg_comparison_summary(
    rdf_artifacts: list[dict[str, Any]],
    lpg_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Side-by-side comparison of RDF vs LPG pipeline outputs.

    Compares:
    - Overall metrics (entity counts, link rates)
    - Per-category breakdown
    - Graph structure differences (triple count vs node/edge count)
    """
    result: dict[str, Any] = {
        "overall": {
            "rdf": _metrics_for_artifacts(rdf_artifacts),
            "lpg": _metrics_for_artifacts(lpg_artifacts),
        },
        "per_category": {},
    }

    # Per-category breakdown
    rdf_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lpg_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for a in rdf_artifacts:
        rdf_by_cat[a.get("category", "unknown")].append(a)
    for a in lpg_artifacts:
        lpg_by_cat[a.get("category", "unknown")].append(a)

    all_categories = sorted(set(list(rdf_by_cat.keys()) + list(lpg_by_cat.keys())))
    for cat in all_categories:
        result["per_category"][cat] = {
            "rdf": _metrics_for_artifacts(rdf_by_cat.get(cat, [])),
            "lpg": _metrics_for_artifacts(lpg_by_cat.get(cat, [])),
        }

    # Graph structure comparison
    rdf_triple_counts = []
    lpg_node_counts = []
    lpg_edge_counts = []

    for a in rdf_artifacts:
        if a.get("status") == "completed":
            rdf_triple_counts.append(len(a.get("graph_preview", {}).get("triples", [])))
    for a in lpg_artifacts:
        if a.get("status") == "completed":
            lpg_node_counts.append(len(a.get("graph_preview", {}).get("nodes", [])))
            lpg_edge_counts.append(len(a.get("graph_preview", {}).get("edges", [])))

    result["graph_structure"] = {
        "rdf_avg_triples": round(sum(rdf_triple_counts) / len(rdf_triple_counts), 1) if rdf_triple_counts else 0,
        "lpg_avg_nodes": round(sum(lpg_node_counts) / len(lpg_node_counts), 1) if lpg_node_counts else 0,
        "lpg_avg_edges": round(sum(lpg_edge_counts) / len(lpg_edge_counts), 1) if lpg_edge_counts else 0,
    }

    return result
