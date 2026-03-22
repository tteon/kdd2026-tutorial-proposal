from __future__ import annotations

import re


_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def _normalize_identifier(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub("_", value.strip())
    normalized = _MULTI_UNDERSCORE_RE.sub("_", normalized).strip("_")
    return normalized


def sanitize_label(value: str, *, default: str = "Entity") -> str:
    normalized = _normalize_identifier(value)
    if not normalized:
        return default
    if normalized[0].isdigit():
        normalized = f"N_{normalized}"
    return normalized


def sanitize_relationship_type(value: str, *, default: str = "RELATED_TO") -> str:
    normalized = _normalize_identifier(value).upper()
    if not normalized:
        return default
    if normalized[0].isdigit():
        normalized = f"REL_{normalized}"
    return normalized


def sanitize_node_id(value: str, *, default: str = "node") -> str:
    normalized = _normalize_identifier(value)
    if not normalized:
        return default
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    return normalized.lower()
