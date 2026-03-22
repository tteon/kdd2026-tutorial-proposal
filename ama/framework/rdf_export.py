from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from framework.fibo_loader import get_fibo_prefixes


DEFAULT_PREFIXES: dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "fibo": "https://spec.edmcouncil.org/fibo/ontology/",
    "ama": "https://example.org/ama/",
    **get_fibo_prefixes(),
}

_CURIE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:[^\s]+$")
_URI_RE = re.compile(r"^https?://[^\s]+$")
_NUMERIC_LITERAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:\s+[A-Za-z%$].*)?$")
_NON_LOCAL_RE = re.compile(r"[^A-Za-z0-9_]+")


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _local_name(value: str) -> str:
    normalized = _NON_LOCAL_RE.sub("_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_").lower()
    if not normalized:
        normalized = "resource"
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    return normalized


def _ama_curie(*, example_id: str, kind: str, value: str) -> str:
    return f"ama:{example_id}_{kind}_{_local_name(value)}"


def _is_curie(value: str, prefixes: dict[str, str]) -> bool:
    if not _CURIE_RE.match(value):
        return False
    prefix = value.split(":", 1)[0]
    return prefix in prefixes


def _format_resource(value: str, *, prefixes: dict[str, str], example_id: str, kind: str) -> str:
    if _URI_RE.match(value):
        return f"<{value}>"
    if _is_curie(value, prefixes):
        return value
    return _ama_curie(example_id=example_id, kind=kind, value=value)


_LITERAL_PREDICATES = {"rdfs:label", "ama:mentionId", "ama:linkStatus", "ama:confidence", "ama:targetLabel", "ama:category", "ama:questionText"}


def _format_object(
    value: str,
    *,
    prefixes: dict[str, str],
    example_id: str,
    known_resources: set[str],
    predicate: str = "",
) -> str:
    # Always emit literals for label-like predicates
    if predicate in _LITERAL_PREDICATES:
        return f"\"{_escape_literal(value)}\""
    if _URI_RE.match(value):
        return f"<{value}>"
    if _is_curie(value, prefixes):
        return value
    if value in known_resources:
        return _ama_curie(example_id=example_id, kind="entity", value=value)
    if _NUMERIC_LITERAL_RE.match(value):
        return f"\"{_escape_literal(value)}\""
    return f"\"{_escape_literal(value)}\""


def render_ttl_from_artifact(artifact: dict[str, Any], *, prefixes: dict[str, str] | None = None) -> str:
    prefix_map = dict(DEFAULT_PREFIXES)
    if prefixes:
        prefix_map.update(prefixes)

    example_id = str(artifact.get("example_id", "example"))
    graph = artifact.get("graph_preview", {})
    triples = graph.get("triples", [])

    lines = [f"@prefix {name}: <{uri}> ." for name, uri in sorted(prefix_map.items())]
    lines.append("")
    lines.append(f"# experiment={artifact.get('experiment_name', '')} example_id={example_id}")

    known_subjects = {str(t.get("subject", "")).strip() for t in triples if str(t.get("subject", "")).strip()}
    linked_names = {
        str(entity.get("canonical_name", "")).strip()
        for entity in artifact.get("linked_entities", [])
        if str(entity.get("canonical_name", "")).strip()
    }
    known_resources = known_subjects | linked_names

    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not subject or not predicate or not obj:
            continue

        s = _format_resource(subject, prefixes=prefix_map, example_id=example_id, kind="entity")
        p = _format_resource(predicate, prefixes=prefix_map, example_id=example_id, kind="predicate")
        o = _format_object(obj, prefixes=prefix_map, example_id=example_id, known_resources=known_resources, predicate=predicate)
        lines.append(f"{s} {p} {o} .")

    lines.append("")
    return "\n".join(lines)


def build_rdf_export_metadata(artifact: dict[str, Any], *, output_path: Path | None = None) -> dict[str, Any]:
    return {
        "export_format": "text/turtle",
        "prefixes": dict(DEFAULT_PREFIXES),
        "subject_strategy": "ama:<example_id>_document and ama:<example_id>_entity_<normalized_local_name>",
        "predicate_strategy": "known CURIE when available, otherwise ama:<example_id>_predicate_<normalized_local_name>",
        "object_strategy": "CURIE/IRI for graph resources, literal for plain text values",
        "materialization_method": "deterministic_projection_from_linked_entities",
        "ttl_path": str(output_path) if output_path else "",
    }


def write_ttl_artifact(
    artifact: dict[str, Any],
    *,
    output_path: Path,
    prefixes: dict[str, str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_ttl_from_artifact(artifact, prefixes=prefixes),
        encoding="utf-8",
    )
    return output_path
