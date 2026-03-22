"""
Evaluation functions for extraction pipeline artifacts.

Four evaluation levels:

  1. evaluate_extraction     — entity P/R/F1 against gold (text match)
  2. evaluate_linking        — linking accuracy against gold (concept_id match)
  3. self_consistency        — metrics without gold (null_link_rate, etc.)
  4. fibo_conformance        — FIBO Ontology Guide compliance (no gold needed)

The FIBO conformance evaluator (Level 4) derives criteria from:
  context-ONTOLOGY_GUIDE.md — FIBO best practices for naming, IRI format,
  definitions, unique labels, and hygiene tests.

This gives a principled, authority-grounded evaluation axis independent of
any manually curated gold standard.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExtractionScores:
    """Precision, recall, F1 for entity extraction (text-level match)."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class LinkingScores:
    """Linking accuracy: fraction of correctly linked entities."""
    accuracy: float = 0.0
    correct: int = 0
    total: int = 0


def evaluate_extraction(
    predicted_entities: list[dict[str, Any]],
    gold_entity_texts: set[str],
) -> ExtractionScores:
    """Compare predicted entity mentions against gold entity texts.

    Match is case-insensitive, whitespace-normalized.
    """
    predicted_texts = set()
    for ent in predicted_entities:
        text = str(ent.get("text", "") or ent.get("canonical_name", "")).strip().lower()
        text = " ".join(text.split())
        if text:
            predicted_texts.add(text)

    gold_normalized = {" ".join(t.strip().lower().split()) for t in gold_entity_texts if t}

    tp = len(predicted_texts & gold_normalized)
    fp = len(predicted_texts - gold_normalized)
    fn = len(gold_normalized - predicted_texts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ExtractionScores(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate_linking(
    predicted_linked: list[dict[str, Any]],
    gold_concept_pairs: set[tuple[str, str]],
) -> LinkingScores:
    """Compare predicted (canonical_name, target_id) against gold pairs.

    Match is case-insensitive on canonical_name, exact on concept_id.
    """
    if not gold_concept_pairs:
        return LinkingScores()

    predicted_pairs = set()
    for le in predicted_linked:
        name = str(le.get("canonical_name", "")).strip().lower()
        target = str(le.get("target_id", ""))
        if name and target:
            predicted_pairs.add((name, target))

    correct = len(predicted_pairs & gold_concept_pairs)
    total = len(gold_concept_pairs)

    return LinkingScores(
        accuracy=round(correct / total, 4) if total > 0 else 0.0,
        correct=correct,
        total=total,
    )


def compute_self_consistency_metrics(artifact: dict[str, Any]) -> dict[str, Optional[float]]:
    """Compute metrics that don't require gold annotations.

    These are useful for observability even before manual gold is created:
    - null_link_rate
    - ambiguous_link_rate
    - duplicate_entity_rate
    - materialization_success_rate
    """
    linked = artifact.get("linked_entities", [])
    total = len(linked)

    if total == 0:
        return {
            "null_link_rate": None,
            "ambiguous_link_rate": None,
            "duplicate_entity_rate": None,
            "materialization_success_rate": 0.0,
        }

    null_count = sum(1 for le in linked if le.get("status") == "null_link")
    ambig_count = sum(1 for le in linked if le.get("status") == "ambiguous")
    canonical_names = [le.get("canonical_name", "") for le in linked]
    unique_count = len(set(canonical_names))

    graph = artifact.get("graph_preview", {})
    has_graph = bool(graph.get("triples") or graph.get("nodes") or graph.get("edges"))

    return {
        "null_link_rate": round(null_count / total, 4),
        "ambiguous_link_rate": round(ambig_count / total, 4),
        "duplicate_entity_rate": round(1.0 - unique_count / total, 4),
        "materialization_success_rate": 1.0 if has_graph else 0.0,
    }


# ---------------------------------------------------------------------------
# Level 4: FIBO Conformance Evaluation (no gold needed)
# ---------------------------------------------------------------------------
#
# Derived from context-ONTOLOGY_GUIDE.md (FIBO best practices):
#
#   - IRI format:   fibo-<domain>-<module>-...-<ontology>:ClassName
#   - Classes:      UpperCamelCase, singular, no abbreviations
#   - Properties:   lowerCamelCase, start with verb
#   - Unique labels: no duplicate concept_ids for different labels
#   - Namespace:    must reference known FIBO domains (BE, FND, FBC, etc.)
#   - RDF triples:  predicates lowerCamelCase verbs, typed subjects/objects
#
# These checks measure "does the pipeline output conform to FIBO standards?"
# rather than "did it extract the right entities?" — no gold needed.
# ---------------------------------------------------------------------------

# Known FIBO domain prefixes (from https://spec.edmcouncil.org/fibo/ontology/)
KNOWN_FIBO_DOMAINS = {
    "fibo-be",    # Business Entities
    "fibo-cae",   # Corporate Actions & Events
    "fibo-der",   # Derivatives
    "fibo-fbc",   # Financial Business & Commerce
    "fibo-fnd",   # Foundations
    "fibo-ind",   # Indices & Indicators
    "fibo-loan",  # Loans
    "fibo-md",    # Market Data
    "fibo-sec",   # Securities
    "fibo-bp",    # Business Process
}

# Regex for FIBO namespace prefix: fibo-<domain>-<module>-...-<ontology>
FIBO_PREFIX_RE = re.compile(r"^fibo(-[a-z]{2,6}){1,5}$")

# Regex for UpperCamelCase (FIBO class naming convention)
UPPER_CAMEL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")

# Regex for lowerCamelCase (FIBO property naming convention)
LOWER_CAMEL_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")

# Common abbreviations that FIBO guide says to avoid in names
ABBREVIATION_PATTERNS = re.compile(
    r"\b(LLC|INC|PLC|AG|SA|NV|CORP|LTD|ETF|OTC|CDO|CDS|ABS|MBS|IPO|CEO|CFO|COO)\b",
    re.IGNORECASE,
)


@dataclass
class FIBOConformanceScores:
    """FIBO Ontology Guide conformance scores for one artifact.

    Each score is a rate (0.0–1.0) measuring the fraction of items
    that pass the corresponding FIBO best-practice check.
    """
    # Entity type naming (§ Classes: UpperCamelCase, singular)
    entity_type_camelcase_rate: float = 0.0
    entity_type_singular_rate: float = 0.0
    entity_type_no_abbreviation_rate: float = 0.0

    # Concept ID format (§ FIBO standard IRI format)
    concept_id_has_prefix_rate: float = 0.0
    concept_id_known_domain_rate: float = 0.0
    concept_id_class_camelcase_rate: float = 0.0

    # Label uniqueness (§ Unique labels)
    concept_label_unique_rate: float = 0.0

    # Namespace coverage (FIBO vs AMA extensions)
    fibo_namespace_rate: float = 0.0  # fraction of linked entities using fibo-* prefix

    # RDF triple hygiene (§ Properties: lowerCamelCase verbs)
    triple_predicate_camelcase_rate: float = 0.0
    triple_predicate_has_prefix_rate: float = 0.0
    triple_subject_typed_rate: float = 0.0
    triple_object_typed_rate: float = 0.0

    # Aggregate
    overall_conformance: float = 0.0
    semantic_conformance: float = 0.0
    structural_conformance: float = 0.0

    # Detail counts
    total_entities: int = 0
    total_linked: int = 0
    total_triples: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    violations: list[str] = field(default_factory=list)


def _safe_rate(passing: int, total: int) -> float:
    return round(passing / total, 4) if total > 0 else 1.0


def _parse_concept_id(concept_id: str) -> tuple[str, str]:
    """Split 'fibo-be-le-lp:LegalEntity' into ('fibo-be-le-lp', 'LegalEntity').

    Also handles 'fibo:Revenue' and full IRIs.
    """
    if ":" in concept_id:
        prefix, _, local = concept_id.partition(":")
        return prefix.strip(), local.strip()
    return "", concept_id.strip()


def _parse_triple_uri(uri: str) -> tuple[str, str]:
    """Split a triple URI like 'fibo-be-le-cb:isDefendantIn' or
    'fibo-fnd-org-fm:Corporation/AbbVie' into (prefix, local_name).
    """
    if ":" in uri:
        prefix, _, rest = uri.partition(":")
        # Handle 'Corporation/AbbVie' → take 'Corporation' as the class part
        local = rest.split("/")[0] if "/" in rest else rest
        return prefix.strip(), local.strip()
    return "", uri.strip()


def _is_singular(word: str) -> bool:
    """Heuristic: a CamelCase class name is likely plural if it ends in 's'
    but not 'ss', 'us', 'is', 'sis', 'ness'.
    """
    if not word:
        return True
    lower = word.lower()
    if lower.endswith(("ss", "us", "is", "sis", "ness", "less", "ous", "ics")):
        return True  # These endings are not plural markers
    if lower.endswith("s") and not lower.endswith("s's"):
        return False  # Likely plural
    return True


def evaluate_fibo_conformance(artifact: dict[str, Any]) -> FIBOConformanceScores:
    """Evaluate how well a pipeline artifact conforms to FIBO best practices.

    Checks entity types, concept IDs, label uniqueness, and RDF triple hygiene
    against the rules in the FIBO Ontology Guide.

    No gold standard needed — this measures structural quality.
    """
    scores = FIBOConformanceScores()
    violations: list[str] = []

    # ----- Entity type checks (extracted_entities) -----
    entities = artifact.get("extracted_entities", [])
    scores.total_entities = len(entities)

    camelcase_ok = 0
    singular_ok = 0
    no_abbrev_ok = 0

    for ent in entities:
        etype = str(ent.get("entity_type", ""))
        if not etype:
            continue

        if UPPER_CAMEL_RE.match(etype):
            camelcase_ok += 1
        else:
            violations.append(f"entity_type '{etype}' is not UpperCamelCase")

        if _is_singular(etype):
            singular_ok += 1
        else:
            violations.append(f"entity_type '{etype}' appears plural (should be singular)")

        if not ABBREVIATION_PATTERNS.search(etype):
            no_abbrev_ok += 1
        else:
            violations.append(f"entity_type '{etype}' contains abbreviation")

    scores.entity_type_camelcase_rate = _safe_rate(camelcase_ok, len(entities))
    scores.entity_type_singular_rate = _safe_rate(singular_ok, len(entities))
    scores.entity_type_no_abbreviation_rate = _safe_rate(no_abbrev_ok, len(entities))

    # ----- Concept ID checks (linked_entities) -----
    linked = artifact.get("linked_entities", [])
    scores.total_linked = len(linked)

    prefix_ok = 0
    domain_ok = 0
    class_camelcase_ok = 0

    label_to_ids: dict[str, set[str]] = {}

    for le in linked:
        concept_id = str(le.get("target_id", ""))
        concept_label = str(le.get("target_label", ""))

        if not concept_id:
            continue

        prefix, local_name = _parse_concept_id(concept_id)

        # Check: has a recognizable prefix
        if prefix and (FIBO_PREFIX_RE.match(prefix) or prefix == "fibo"):
            prefix_ok += 1
        else:
            violations.append(f"concept_id '{concept_id}' has non-FIBO prefix '{prefix}'")

        # Check: known domain
        domain_part = "-".join(prefix.split("-")[:2]) if "-" in prefix else prefix
        if domain_part in KNOWN_FIBO_DOMAINS or prefix == "fibo":
            domain_ok += 1
        else:
            violations.append(f"concept_id '{concept_id}' references unknown domain '{domain_part}'")

        # Check: local name is UpperCamelCase
        if local_name and UPPER_CAMEL_RE.match(local_name):
            class_camelcase_ok += 1
        else:
            violations.append(f"concept_id '{concept_id}' local name '{local_name}' is not UpperCamelCase")

        # Track label uniqueness
        if concept_label:
            label_to_ids.setdefault(concept_label, set()).add(concept_id)

    scores.concept_id_has_prefix_rate = _safe_rate(prefix_ok, len(linked))
    scores.concept_id_known_domain_rate = _safe_rate(domain_ok, len(linked))
    scores.concept_id_class_camelcase_rate = _safe_rate(class_camelcase_ok, len(linked))

    # FIBO namespace rate: fraction using actual fibo-* prefix (vs ama: extensions)
    fibo_ns_count = sum(
        1 for le in linked
        if str(le.get("target_id", "")).startswith("fibo")
    )
    scores.fibo_namespace_rate = _safe_rate(fibo_ns_count, len(linked))

    # Label uniqueness: fraction of labels that map to exactly one concept_id
    if label_to_ids:
        unique_labels = sum(1 for ids in label_to_ids.values() if len(ids) == 1)
        scores.concept_label_unique_rate = _safe_rate(unique_labels, len(label_to_ids))
        for label, ids in label_to_ids.items():
            if len(ids) > 1:
                violations.append(f"label '{label}' maps to multiple concept_ids: {ids}")
    else:
        scores.concept_label_unique_rate = 1.0

    # ----- RDF triple hygiene (graph_preview.triples) -----
    graph = artifact.get("graph_preview", {})
    triples = graph.get("triples", [])
    scores.total_triples = len(triples)
    scores.total_nodes = len(graph.get("nodes", []))
    scores.total_edges = len(graph.get("edges", []))

    pred_camelcase_ok = 0
    pred_prefix_ok = 0
    subj_typed_ok = 0
    obj_typed_ok = 0

    for triple in triples:
        subj = str(triple.get("subject", ""))
        pred = str(triple.get("predicate", ""))
        obj = str(triple.get("object", ""))

        # Predicate: should be lowerCamelCase with FIBO prefix
        pred_prefix, pred_local = _parse_triple_uri(pred)
        if pred_local and LOWER_CAMEL_RE.match(pred_local):
            pred_camelcase_ok += 1
        else:
            violations.append(f"predicate '{pred}' local name '{pred_local}' is not lowerCamelCase")

        if pred_prefix and (FIBO_PREFIX_RE.match(pred_prefix) or pred_prefix == "fibo"):
            pred_prefix_ok += 1
        else:
            violations.append(f"predicate '{pred}' has non-FIBO prefix '{pred_prefix}'")

        # Subject: should be typed (has a prefix:Class/instance pattern)
        subj_prefix, _ = _parse_triple_uri(subj)
        if subj_prefix and (FIBO_PREFIX_RE.match(subj_prefix) or subj_prefix == "fibo"):
            subj_typed_ok += 1
        else:
            violations.append(f"subject '{subj}' appears untyped (no FIBO prefix)")

        # Object: typed if it has FIBO prefix, or it's a literal (acceptable)
        obj_prefix, _ = _parse_triple_uri(obj)
        if obj_prefix and (FIBO_PREFIX_RE.match(obj_prefix) or obj_prefix == "fibo"):
            obj_typed_ok += 1
        elif ":" not in obj:
            obj_typed_ok += 1  # Literal value — acceptable per FIBO
        else:
            violations.append(f"object '{obj}' has non-FIBO prefix '{obj_prefix}'")

    scores.triple_predicate_camelcase_rate = _safe_rate(pred_camelcase_ok, len(triples))
    scores.triple_predicate_has_prefix_rate = _safe_rate(pred_prefix_ok, len(triples))
    scores.triple_subject_typed_rate = _safe_rate(subj_typed_ok, len(triples))
    scores.triple_object_typed_rate = _safe_rate(obj_typed_ok, len(triples))

    # ----- LPG edge type check (for LPG artifacts) -----
    # LPG edge types are SCREAMING_SNAKE_CASE by convention, not FIBO-standard,
    # so we don't penalize them here. But we do check node labels.
    nodes = graph.get("nodes", [])
    for node in nodes:
        labels = node.get("labels", [])
        for label in labels:
            if not UPPER_CAMEL_RE.match(label):
                violations.append(f"LPG node label '{label}' is not UpperCamelCase")

    # ----- Conformance scores (overall, semantic, structural) -----
    # Semantic: entity type + concept ID + label dimensions (representation-agnostic)
    semantic_components: list[tuple[float, float]] = [
        (scores.entity_type_camelcase_rate, 1.0),
        (scores.entity_type_singular_rate, 0.5),
        (scores.entity_type_no_abbreviation_rate, 0.5),
        (scores.concept_id_has_prefix_rate, 1.0),
        (scores.concept_id_known_domain_rate, 1.0),
        (scores.concept_id_class_camelcase_rate, 1.0),
        (scores.concept_label_unique_rate, 0.5),
    ]

    # Structural: triple hygiene dimensions (RDF-only, always 0/1 for LPG)
    structural_components: list[tuple[float, float]] = []
    if triples:
        structural_components = [
            (scores.triple_predicate_camelcase_rate, 1.0),
            (scores.triple_predicate_has_prefix_rate, 1.0),
            (scores.triple_subject_typed_rate, 1.0),
            (scores.triple_object_typed_rate, 0.5),
        ]

    sem_weight = sum(w for _, w in semantic_components)
    sem_sum = sum(s * w for s, w in semantic_components)
    scores.semantic_conformance = round(sem_sum / sem_weight, 4) if sem_weight > 0 else 0.0

    if structural_components:
        str_weight = sum(w for _, w in structural_components)
        str_sum = sum(s * w for s, w in structural_components)
        scores.structural_conformance = round(str_sum / str_weight, 4) if str_weight > 0 else 0.0
    else:
        scores.structural_conformance = float("nan")  # N/A for LPG

    # Overall = all applicable dimensions combined
    all_components = semantic_components + structural_components
    total_weight = sum(w for _, w in all_components)
    weighted_sum = sum(score * w for score, w in all_components)
    scores.overall_conformance = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    scores.violations = violations
    return scores


def aggregate_conformance(
    conformance_list: list[FIBOConformanceScores],
) -> dict[str, Any]:
    """Macro-average FIBO conformance scores across multiple artifacts."""
    n = len(conformance_list)
    if n == 0:
        return {"count": 0}

    fields = [
        "entity_type_camelcase_rate", "entity_type_singular_rate",
        "entity_type_no_abbreviation_rate",
        "concept_id_has_prefix_rate", "concept_id_known_domain_rate",
        "fibo_namespace_rate",
        "concept_id_class_camelcase_rate", "concept_label_unique_rate",
        "triple_predicate_camelcase_rate", "triple_predicate_has_prefix_rate",
        "triple_subject_typed_rate", "triple_object_typed_rate",
        "overall_conformance", "semantic_conformance", "structural_conformance",
    ]

    import math

    result: dict[str, Any] = {"count": n}
    for f in fields:
        vals = [getattr(s, f) for s in conformance_list]
        # Filter NaN values (structural_conformance is NaN for LPG)
        valid = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if valid:
            result[f"macro_{f}"] = round(sum(valid) / len(valid), 4)
        else:
            result[f"macro_{f}"] = None

    total_violations = sum(len(s.violations) for s in conformance_list)
    result["total_violations"] = total_violations
    result["avg_violations_per_artifact"] = round(total_violations / n, 1)

    return result


# ---------------------------------------------------------------------------
# Aggregation (gold-based)
# ---------------------------------------------------------------------------

def aggregate_metrics(
    extraction_scores: list[ExtractionScores],
    linking_scores: list[LinkingScores],
) -> dict[str, Any]:
    """Aggregate metrics across multiple examples (macro averages)."""
    n = len(extraction_scores)
    if n == 0:
        return {"extraction": {}, "linking": {}, "count": 0}

    avg_precision = sum(s.precision for s in extraction_scores) / n
    avg_recall = sum(s.recall for s in extraction_scores) / n
    avg_f1 = sum(s.f1 for s in extraction_scores) / n

    ln = len(linking_scores)
    avg_linking_acc = sum(s.accuracy for s in linking_scores) / ln if ln > 0 else 0.0

    return {
        "extraction": {
            "macro_precision": round(avg_precision, 4),
            "macro_recall": round(avg_recall, 4),
            "macro_f1": round(avg_f1, 4),
        },
        "linking": {
            "macro_accuracy": round(avg_linking_acc, 4),
        },
        "count": n,
    }


# ---------------------------------------------------------------------------
# Opik BaseMetric subclasses for evaluate() integration
# ---------------------------------------------------------------------------

try:
    from opik.evaluation.metrics.base_metric import BaseMetric
    from opik.evaluation.metrics.score_result import ScoreResult

    _HAS_OPIK = True
except ImportError:
    _HAS_OPIK = False
    BaseMetric = object  # type: ignore[misc,assignment]
    ScoreResult = None  # type: ignore[misc,assignment]


class FIBOConformanceMetric(BaseMetric):  # type: ignore[misc]
    """Opik metric: FIBO Ontology Guide conformance scores.

    Returns multiple ScoreResults covering semantic, structural,
    and overall conformance dimensions. No gold needed.
    """

    def __init__(self) -> None:
        if _HAS_OPIK:
            super().__init__(name="fibo_conformance")

    def score(self, artifact: dict[str, Any], **ignored: Any) -> list:
        conf = evaluate_fibo_conformance(artifact)
        results = [
            ScoreResult(
                name="fibo_overall",
                value=conf.overall_conformance,
                reason=f"{len(conf.violations)} violations",
                metadata={"total_entities": conf.total_entities,
                          "total_linked": conf.total_linked,
                          "total_triples": conf.total_triples},
            ),
            ScoreResult(
                name="fibo_semantic",
                value=conf.semantic_conformance,
                category_name="semantic",
            ),
            ScoreResult(
                name="fibo_namespace_rate",
                value=conf.fibo_namespace_rate,
                category_name="semantic",
            ),
            ScoreResult(
                name="fibo_entity_type_camelcase",
                value=conf.entity_type_camelcase_rate,
                category_name="semantic",
            ),
            ScoreResult(
                name="fibo_concept_id_prefix",
                value=conf.concept_id_has_prefix_rate,
                category_name="semantic",
            ),
            ScoreResult(
                name="fibo_concept_id_domain",
                value=conf.concept_id_known_domain_rate,
                category_name="semantic",
            ),
            ScoreResult(
                name="fibo_label_unique",
                value=conf.concept_label_unique_rate,
                category_name="semantic",
            ),
        ]
        # Structural (RDF-only; NaN for LPG)
        if not math.isnan(conf.structural_conformance):
            results.append(ScoreResult(
                name="fibo_structural",
                value=conf.structural_conformance,
                category_name="structural",
            ))
            results.append(ScoreResult(
                name="fibo_pred_camelcase",
                value=conf.triple_predicate_camelcase_rate,
                category_name="structural",
            ))
            results.append(ScoreResult(
                name="fibo_pred_prefix",
                value=conf.triple_predicate_has_prefix_rate,
                category_name="structural",
            ))
        return results


class SelfConsistencyMetric(BaseMetric):  # type: ignore[misc]
    """Opik metric: self-consistency checks (no gold needed).

    null_link_rate, ambiguous_link_rate, duplicate_entity_rate,
    materialization_success_rate.
    """

    def __init__(self) -> None:
        if _HAS_OPIK:
            super().__init__(name="self_consistency")

    def score(self, artifact: dict[str, Any], **ignored: Any) -> list:
        metrics = compute_self_consistency_metrics(artifact)
        results = []
        for key, value in metrics.items():
            if value is not None:
                results.append(ScoreResult(
                    name=f"sc_{key}",
                    value=value,
                    category_name="self_consistency",
                ))
        return results


class GoldExtractionMetric(BaseMetric):  # type: ignore[misc]
    """Opik metric: entity extraction P/R/F1 against gold annotations."""

    def __init__(self) -> None:
        if _HAS_OPIK:
            super().__init__(name="gold_extraction")

    def score(
        self,
        artifact: dict[str, Any],
        gold_entity_texts: list[str] | None = None,
        **ignored: Any,
    ) -> list:
        if not gold_entity_texts:
            return [ScoreResult(name="gold_f1", value=0.0,
                                reason="no gold", scoring_failed=True)]
        gold_set = set(gold_entity_texts)
        predicted = artifact.get("extracted_entities", [])
        ext = evaluate_extraction(predicted, gold_set)
        return [
            ScoreResult(name="gold_precision", value=ext.precision,
                        category_name="gold"),
            ScoreResult(name="gold_recall", value=ext.recall,
                        category_name="gold"),
            ScoreResult(name="gold_f1", value=ext.f1,
                        category_name="gold",
                        metadata={"tp": ext.true_positives,
                                  "fp": ext.false_positives,
                                  "fn": ext.false_negatives}),
        ]


class GoldLinkingMetric(BaseMetric):  # type: ignore[misc]
    """Opik metric: linking accuracy against gold concept pairs."""

    def __init__(self) -> None:
        if _HAS_OPIK:
            super().__init__(name="gold_linking")

    def score(
        self,
        artifact: dict[str, Any],
        gold_concept_pairs: list[list[str]] | None = None,
        **ignored: Any,
    ) -> list:
        if not gold_concept_pairs:
            return [ScoreResult(name="gold_link_accuracy", value=0.0,
                                reason="no gold", scoring_failed=True)]
        gold_set = {(p[0], p[1]) for p in gold_concept_pairs}
        predicted = artifact.get("linked_entities", [])
        link = evaluate_linking(predicted, gold_set)
        return [
            ScoreResult(name="gold_link_accuracy", value=link.accuracy,
                        category_name="gold",
                        metadata={"correct": link.correct,
                                  "total": link.total}),
        ]


# ---------------------------------------------------------------------------
# Opik evaluate() runner
# ---------------------------------------------------------------------------

def _load_artifact(path: str) -> dict[str, Any]:
    """Load a single artifact JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_opik_evaluate(
    experiment_name: str,
    manifest: dict[str, Any],
    gold_by_id: dict[str, Any] | None = None,
    project_name: str = "ama_evaluation",
) -> Any:
    """Run opik.evaluate() over experiment artifacts.

    Creates (or reuses) an Opik Dataset from the manifest, defines a task
    that loads each artifact, and scores with FIBO conformance +
    self-consistency + gold metrics (when available).

    Returns the EvaluationResult for further processing.
    """
    if not _HAS_OPIK:
        raise ImportError("opik is required for run_opik_evaluate()")

    import opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics.score_result import ScoreResult as SR

    client = opik.Opik()

    # --- Build dataset items from manifest ---
    ontology_mode = manifest.get("experiment", {}).get("ontology_mode", "all_packs")
    model = manifest.get("model", "unknown")
    dataset_name = f"ama_eval_{experiment_name}"

    items: list[dict[str, Any]] = []
    for entry in manifest.get("examples", []):
        example_id = entry["example_id"]
        rep = entry["representation"]
        category = entry.get("category", "")

        item: dict[str, Any] = {
            "example_id": example_id,
            "representation": rep,
            "category": category,
            "artifact_path": entry["artifact_path"],
            "ontology_mode": entry.get("ontology_mode", ontology_mode),
        }

        # Attach gold data if available
        if gold_by_id and example_id in gold_by_id:
            gold = gold_by_id[example_id]
            item["gold_entity_texts"] = list(gold.entity_texts())
            item["gold_concept_pairs"] = [
                [name, cid] for name, cid in gold.entity_concept_pairs()
            ]

        items.append(item)

    # Create or get dataset
    dataset = client.get_or_create_dataset(name=dataset_name)
    dataset.insert(items)

    # --- Task: load artifact from disk ---
    def task(item: dict[str, Any]) -> dict[str, Any]:
        artifact = _load_artifact(item["artifact_path"])
        return {"artifact": artifact}

    # --- Metrics ---
    metrics = [
        FIBOConformanceMetric(),
        SelfConsistencyMetric(),
    ]
    if gold_by_id:
        metrics.append(GoldExtractionMetric())
        metrics.append(GoldLinkingMetric())

    # --- Experiment-level scoring functions ---
    def macro_fibo_overall(test_results: list) -> SR:
        vals = []
        for tr in test_results:
            for sr in tr.score_results:
                if sr.name == "fibo_overall":
                    vals.append(sr.value)
        avg = sum(vals) / len(vals) if vals else 0.0
        return SR(name="macro_fibo_overall", value=round(avg, 4),
                  reason=f"avg over {len(vals)} items")

    def macro_gold_f1(test_results: list) -> SR:
        vals = []
        for tr in test_results:
            for sr in tr.score_results:
                if sr.name == "gold_f1" and not sr.scoring_failed:
                    vals.append(sr.value)
        avg = sum(vals) / len(vals) if vals else 0.0
        return SR(name="macro_gold_f1", value=round(avg, 4),
                  reason=f"avg over {len(vals)} items")

    experiment_scoring_fns = [macro_fibo_overall]
    if gold_by_id:
        experiment_scoring_fns.append(macro_gold_f1)

    # --- Run ---
    result = evaluate(
        dataset=dataset,
        task=task,
        scoring_metrics=metrics,
        experiment_name=f"{experiment_name}_eval",
        project_name=project_name,
        experiment_config={
            "experiment_name": experiment_name,
            "ontology_mode": ontology_mode,
            "model": model,
            "sample_size": manifest.get("sampled_example_count", 0),
        },
        experiment_scoring_functions=experiment_scoring_fns,
        experiment_tags=[experiment_name, ontology_mode],
    )

    return result
