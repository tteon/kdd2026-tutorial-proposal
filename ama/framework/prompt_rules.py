from __future__ import annotations

from typing import Dict

from framework.fibo_loader import FIBOConceptSpec, load_fibo_specs
from framework.schema import OntologySelection, PromptContext, PromptRule

# Re-export for backward compatibility
__all__ = ["FIBOConceptSpec", "FIBO_CONCEPT_SPECS"]

FIBO_CONCEPT_SPECS: Dict[str, FIBOConceptSpec] = load_fibo_specs()


def _fallback_rule(label: str) -> PromptRule:
    return PromptRule(
        concept_id="fibo:UnknownConcept",
        label=label,
        definition="A finance-related concept selected from the ontology context.",
        synonyms=[label],
        semantic_features=["finance context"],
        soft_signals=["the text semantically aligns with the ontology-selected concept"],
        extraction_hint="If the text strongly suggests this concept, classify it here; otherwise return uncertain.",
        linking_hint="Prefer the most specific matching concept when the label, definition, and context align.",
    )


def rule_from_spec(spec: FIBOConceptSpec, *, bare: bool = False) -> PromptRule:
    if bare:
        # Condition B: concept ID + label only, no metadata
        return PromptRule(
            concept_id=spec.concept_id,
            label=spec.label,
            definition="",
            synonyms=[],
            semantic_features=[],
            soft_signals=[],
            extraction_hint=f"Classify as {spec.label} if the text clearly refers to this concept.",
            linking_hint=f"Link to {spec.label} when the entity mention matches this concept.",
        )
    return PromptRule(
        concept_id=spec.concept_id,
        label=spec.label,
        definition=spec.definition,
        synonyms=list(spec.synonyms),
        semantic_features=list(spec.semantic_features),
        soft_signals=list(spec.soft_signals),
        extraction_hint=(
            "If the text suggests the concept via its semantic features and context, classify it as "
            f"{spec.label}; if evidence is weak, return uncertain."
        ),
        linking_hint=(
            "Match the entity to this concept when the mention, synonyms, and surrounding context align; "
            "prefer this concept over broader parents only when the evidence is specific enough."
        ),
    )


def build_prompt_rules(selection: OntologySelection) -> list[PromptRule]:
    bare = selection.selection_mode == "rule_only"
    rules: list[PromptRule] = []
    for class_name in selection.selected_classes:
        spec = FIBO_CONCEPT_SPECS.get(class_name)
        rules.append(rule_from_spec(spec, bare=bare) if spec else _fallback_rule(class_name))
    return rules


def build_extraction_prompt(*, selection: OntologySelection, rules: list[PromptRule]) -> str:
    lines = [
        "You are an ontology-guided financial entity extraction system.",
        "",
        "Task:",
        "- Read the text and extract financial entities grounded in the selected ontology context.",
        "- Use semantic hints, not strict symbolic logic.",
        "- If evidence is weak or ambiguous, return uncertain instead of over-committing.",
        "",
        "Selected ontology modules:",
    ]
    for module_name in selection.selected_modules:
        lines.append(f"- {module_name}")

    lines.extend(["", "Ontology-guided rules:"])
    for index, rule in enumerate(rules, start=1):
        lines.append(f"{index}. {rule.label}")
        if rule.definition:
            lines.append(f"   Definition: {rule.definition}")
        if rule.synonyms:
            lines.append(f"   Synonyms: {', '.join(rule.synonyms)}")
        if rule.semantic_features:
            lines.append(f"   Semantic features: {', '.join(rule.semantic_features)}")
        if rule.soft_signals:
            lines.append(f"   Soft signals: {', '.join(rule.soft_signals)}")
        lines.append(f"   Extraction hint: {rule.extraction_hint}")

    lines.extend(
        [
            "",
            "Output JSON array format:",
            '[{"text": "...", "type": "...", "confidence": 0.0, "evidence": ["..."], "status": "linked|uncertain"}]',
        ]
    )
    return "\n".join(lines)


def build_linking_prompt(*, rules: list[PromptRule]) -> str:
    fibo_rules = [r for r in rules if r.concept_id.startswith("fibo")]
    ama_rules = [r for r in rules if not r.concept_id.startswith("fibo")]

    lines = [
        "You are an ontology-guided entity linking system.",
        "",
        "Task:",
        "- Map each extracted entity to the BEST MATCHING candidate concept from the list below.",
        "- You MUST ONLY use concept_id values from the candidate list below. Do NOT invent or hallucinate concept IDs.",
        "- PREFER FIBO-standard concepts (fibo-* prefix) over AMA-extension concepts (ama: prefix).",
        "- Use an ama: concept ONLY when no FIBO concept is a reasonable match.",
        "- If no candidate concept fits the entity, set status to 'null_link' and concept_id to empty string.",
        "- If multiple candidates fit, pick the best one and list alternatives in ambiguity_candidates.",
        "",
        "IMPORTANT CONSTRAINT: The concept_id in your output MUST be one of these exact values:",
        "",
        "  FIBO-standard concepts (preferred):",
    ]
    for rule in fibo_rules:
        lines.append(f"  - {rule.concept_id}")

    if ama_rules:
        lines.append("")
        lines.append("  AMA-extension concepts (use only when no FIBO concept fits):")
        for rule in ama_rules:
            lines.append(f"  - {rule.concept_id}")

    lines.extend(["", "Candidate concepts (full details):"])
    for index, rule in enumerate(rules, start=1):
        lines.append(f"{index}. {rule.label} ({rule.concept_id})")
        if rule.definition:
            lines.append(f"   Definition: {rule.definition}")
        if rule.synonyms:
            lines.append(f"   Synonyms: {', '.join(rule.synonyms)}")
        lines.append(f"   Linking hint: {rule.linking_hint}")

    lines.extend(
        [
            "",
            "If NONE of the above concepts match an entity, use status='null_link' and concept_id=''.",
            "",
            "Output JSON array format:",
            '[{"mention_id": "m-000", "mention_text": "...", "concept_id": "...", "concept_label": "...", "confidence": 0.0, "status": "linked|ambiguous|null_link", "ambiguity_candidates": []}]',
        ]
    )
    return "\n".join(lines)


def build_relation_extraction_prompt(*, rules: list[PromptRule]) -> str:
    """Build a prompt for extracting semantic relations between entities."""
    concept_labels = [rule.label for rule in rules]
    lines = [
        "You are an ontology-guided relation extraction system for financial documents.",
        "",
        "Task:",
        "- Given a list of extracted entities (with mention_ids) and the source text,",
        "  identify semantic relations BETWEEN the entities.",
        "- Each relation has a source entity, a target entity, and a relation type.",
        "- Relation types MUST be in lowerCamelCase (e.g., hasRevenue, isSubsidiaryOf, isRegulatedBy).",
        "- Only extract relations that are explicitly supported by the text.",
        "- If no meaningful relations exist between entities, return an empty list.",
        "",
        "Common financial relation types (use these when applicable, or create precise lowerCamelCase types):",
        "  - hasRevenue, hasExpense, hasIncome — entity to monetary amount",
        "  - isSubsidiaryOf, isParentOf — corporate hierarchy",
        "  - isRegulatedBy, regulatesEntity — regulatory relationships",
        "  - operatesIn — entity to geographic region",
        "  - hasSegment — entity to business segment",
        "  - hasDividend, hasStockRepurchase — shareholder return relations",
        "  - isPartyTo — entity to legal proceeding",
        "  - compliesWith — entity to compliance standard",
        "  - appointedBy, oversees — governance relations",
        "  - hasAccountingPolicy — entity to accounting policy",
        "",
        f"Entity types in scope: {', '.join(concept_labels)}",
        "",
        "Output JSON array format:",
        '[{"source_mention_id": "m-000", "target_mention_id": "m-001", "relation_type": "hasRevenue", "confidence": 0.85, "evidence": "..."}]',
    ]
    return "\n".join(lines)


def build_prompt_context(selection: OntologySelection) -> PromptContext:
    rules = build_prompt_rules(selection)
    return PromptContext(
        rules=rules,
        extraction_prompt_preview=build_extraction_prompt(selection=selection, rules=rules),
        linking_prompt_preview=build_linking_prompt(rules=rules),
        relation_extraction_prompt_preview=build_relation_extraction_prompt(rules=rules),
        notes=[
            "Rules are semantic hints translated from ontology intent, not direct OWL axioms.",
            "Use these prompts as inspectable scaffolding before adding automatic OWL parsing.",
        ],
    )
