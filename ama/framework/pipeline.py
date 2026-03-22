"""
Extraction pipeline — one function per experiment stage.

Pipeline flow (each step is a separate function, traced individually):

  make_base_artifact      → create artifact shell with config/metadata
  join_references         → combine references list into extraction text
  extract_entities        → LLM call: text → entity mentions
  normalize_entities      → deterministic: lowercase, strip, dedup
  link_entities           → LLM call: mentions → FIBO concept links (with hallucination guard)
  extract_relations       → LLM call: entities + text → semantic relations
  materialize_graph       → deterministic: entities + relations → RDF triples or LPG nodes/edges
  compute_basic_metrics   → self-consistency metrics (no gold needed)

The orchestrator `run_extraction_pipeline` calls these in order,
wrapping each in a tracer span for Opik visibility.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter
from typing import Any, Union
from uuid import uuid4

from openai import OpenAI

from framework.llm_client import structured_completion
from framework.llm_models import (
    ExtractionResponse,
    LinkingResponse,
    RelationExtractionResponse,
)
from framework.neo4j_safe import sanitize_label, sanitize_node_id, sanitize_relationship_type
from framework.ontology import select_ontology_context
from framework.prompt_rules import build_prompt_context
from framework.schema import (
    EntityMention,
    ExampleArtifact,
    ExampleMetrics,
    ExperimentConfig,
    GraphPreview,
    LinkedEntity,
    NormalizedEntity,
    OntologySelection,
    PromptContext,
)


# ---------------------------------------------------------------------------
# Opik Prompt registration (auto-versioned, cached per content hash)
# ---------------------------------------------------------------------------

_prompt_commit_cache: dict[str, str] = {}  # prompt_hash → commit


def _register_opik_prompt(name: str, prompt_text: str, *, ontology_mode: str) -> str:
    """Register a prompt with Opik Cloud. Returns commit hash, or '' on failure."""
    cache_key = _prompt_hash(prompt_text)
    if cache_key in _prompt_commit_cache:
        return _prompt_commit_cache[cache_key]

    if not os.getenv("OPIK_API_KEY"):
        return ""

    try:
        import opik
        p = opik.Prompt(
            name=name,
            prompt=prompt_text,
            metadata={"ontology_mode": ontology_mode},
            validate_placeholders=False,
        )
        _prompt_commit_cache[cache_key] = p.commit
        return p.commit
    except Exception:
        return ""


def _mention_sample(mentions: list[EntityMention], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "mention_id": mention.mention_id,
            "text": mention.text,
            "entity_type": mention.entity_type,
            "confidence": mention.confidence,
        }
        for mention in mentions[:limit]
    ]


def _linked_sample(linked_entities: list[LinkedEntity], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "mention_id": linked.mention_id,
            "canonical_name": linked.canonical_name,
            "target_id": linked.target_id,
            "target_label": linked.target_label,
            "status": linked.status,
            "confidence": linked.confidence,
        }
        for linked in linked_entities[:limit]
    ]


def _prompt_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def prompt_trace_summary(artifact: ExampleArtifact) -> dict[str, Any]:
    selection_prompt = artifact.ontology_selection.selection_prompt_preview
    extraction_prompt = artifact.prompt_context.extraction_prompt_preview
    linking_prompt = artifact.prompt_context.linking_prompt_preview
    relation_prompt = artifact.prompt_context.relation_extraction_prompt_preview
    return {
        "ontology_mode": artifact.ontology_selection.selection_mode,
        "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
        "prompt_template_id": artifact.ontology_selection.prompt_template_id,
        "prompt_rule_count": len(artifact.prompt_context.rules),
        "prompt_rules": [
            {
                "concept_id": rule.concept_id,
                "label": rule.label,
                "synonyms": rule.synonyms[:3],
                "semantic_features": rule.semantic_features[:3],
            }
            for rule in artifact.prompt_context.rules[:8]
        ],
        "selection_prompt_hash": _prompt_hash(selection_prompt),
        "extraction_prompt_hash": _prompt_hash(extraction_prompt),
        "linking_prompt_hash": _prompt_hash(linking_prompt),
        "relation_prompt_hash": _prompt_hash(relation_prompt),
        "selection_prompt_preview": selection_prompt[:500],
        "extraction_prompt_preview": extraction_prompt[:500],
        "linking_prompt_preview": linking_prompt[:500],
        "relation_prompt_preview": relation_prompt[:500],
    }


# ---------------------------------------------------------------------------
# Stage 0: make_base_artifact (metadata shell)
# ---------------------------------------------------------------------------

def make_base_artifact(
    *,
    config: ExperimentConfig,
    example: dict[str, Any],
    representation: str,
    ontology_selection: OntologySelection | None = None,
    prompt_context: PromptContext | None = None,
) -> ExampleArtifact:
    """Create an artifact with config, example metadata, ontology, and prompt context filled in."""
    category = str(example.get(config.category_field, "unknown"))
    example_id = str(example.get(config.id_field) or example.get("example_id") or "")
    document_id = str(example.get("document_id") or example.get("doc_id") or example_id)
    question = str(example.get(config.question_field, ""))
    answer = example.get(config.answer_field)
    if ontology_selection is None:
        ontology_selection = select_ontology_context(
            category=category,
            prompt_template_id=config.prompt_template_id,
            mode=config.ontology_mode,
            question=question,
        )
    if prompt_context is None:
        prompt_context = build_prompt_context(ontology_selection)

    return ExampleArtifact(
        experiment_name=config.experiment_name,
        experiment_version=config.experiment_version,
        representation=representation,
        example_id=example_id,
        document_id=document_id,
        category=category,
        source_dataset=config.dataset_id,
        source_split=config.dataset_split,
        question=question,
        ontology_selection=ontology_selection,
        prompt_context=prompt_context,
        ground_truth={"answer": answer} if answer is not None else {},
    )


# ---------------------------------------------------------------------------
# Stage 1: join_references (prepare extraction text)
# ---------------------------------------------------------------------------

def join_references(example: dict[str, Any], document_field: str) -> str:
    """Combine the references list into a single string for extraction.

    FinDER stores evidence as `references: list[str]`.
    If it's already a string, return as-is.
    """
    raw = example.get(document_field, "")
    if isinstance(raw, list):
        return "\n\n".join(str(r) for r in raw if r)
    return str(raw)


# ---------------------------------------------------------------------------
# Stage 2: extract_entities (LLM call)
# ---------------------------------------------------------------------------

def extract_entities(
    client: OpenAI,
    *,
    model: str,
    extraction_prompt: str,
    text: str,
) -> tuple[list[EntityMention], dict[str, int]]:
    """Call OpenAI to extract entity mentions from the input text.

    Returns (entity mentions, token usage dict).
    """
    result = structured_completion(
        client,
        model=model,
        system_prompt=extraction_prompt,
        user_prompt=f"Extract financial entities from the following text:\n\n{text}",
        response_model=ExtractionResponse,
    )

    mentions: list[EntityMention] = []
    for i, ent in enumerate(result.parsed.entities):
        mentions.append(EntityMention(
            mention_id=f"m-{i:03d}",
            text=ent.text,
            entity_type=ent.entity_type,
            confidence=ent.confidence,
            evidence=ent.evidence,
        ))
    return mentions, result.usage


# ---------------------------------------------------------------------------
# Stage 3: normalize_entities (deterministic, no LLM)
# ---------------------------------------------------------------------------

def normalize_entities(mentions: list[EntityMention]) -> list[NormalizedEntity]:
    """Deterministic normalization: lowercase, strip whitespace, collapse duplicates.

    No LLM call — this is a reproducible text-processing step.
    """
    normalized: list[NormalizedEntity] = []
    for mention in mentions:
        canonical = mention.text.strip().lower()
        # Remove extra internal whitespace
        canonical = " ".join(canonical.split())
        normalized.append(NormalizedEntity(
            mention_id=mention.mention_id,
            canonical_name=canonical,
            normalized_type=mention.entity_type,
            normalization_method="lowercase_strip",
            confidence=mention.confidence,
        ))
    return normalized


# ---------------------------------------------------------------------------
# Stage 4: link_entities (LLM call)
# ---------------------------------------------------------------------------

def _extract_valid_concept_ids(linking_prompt: str) -> set[str]:
    """Parse the linking prompt to extract the set of allowed concept IDs."""
    valid_ids: set[str] = set()
    for line in linking_prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped[2:]:
            candidate = stripped[2:].strip()
            # Concept IDs look like fibo:LegalEntity or fibo-fnd-acc-aeq:AccountingPolicy
            if candidate and " " not in candidate:
                valid_ids.add(candidate)
    return valid_ids


def link_entities(
    client: OpenAI,
    *,
    model: str,
    linking_prompt: str,
    mentions: list[EntityMention],
) -> tuple[list[LinkedEntity], dict[str, int], dict[str, Any]]:
    """Call OpenAI to link extracted entities to FIBO concepts.

    Returns (linked entities, token usage, hallucination_stats).
    hallucination_stats includes reject_count and rejected_ids.
    """
    valid_concept_ids = _extract_valid_concept_ids(linking_prompt)

    mention_descriptions = "\n".join(
        f"- mention_id={m.mention_id} | text=\"{m.text}\" | type={m.entity_type} | confidence={m.confidence}"
        for m in mentions
    )

    result = structured_completion(
        client,
        model=model,
        system_prompt=linking_prompt,
        user_prompt=f"Link these extracted entities to ontology concepts:\n\n{mention_descriptions}",
        response_model=LinkingResponse,
    )

    linked: list[LinkedEntity] = []
    mention_lookup: dict[str, EntityMention] = {}
    for m in mentions:
        mention_lookup[m.mention_id] = m

    hallucination_rejected: list[str] = []

    for le in result.parsed.linked_entities:
        mention = mention_lookup.get(le.mention_id)
        mention_id = mention.mention_id if mention else f"m-unmatched-{uuid4().hex[:6]}"
        canonical_name = mention.text.strip().lower() if mention else le.mention_text.strip().lower()
        notes: list[str] = [] if mention else ["Linking output referenced an unknown mention_id."]

        concept_id = le.concept_id
        status = le.status

        # Hallucination guard: reject concept_ids not in candidate set
        if concept_id and valid_concept_ids and concept_id not in valid_concept_ids:
            notes.append(f"Hallucinated concept_id '{concept_id}' not in candidate set; forced to null_link.")
            hallucination_rejected.append(concept_id)
            concept_id = ""
            status = "null_link"

        # Determine namespace from concept_id prefix
        namespace = "fibo" if concept_id.startswith("fibo") else ("ama" if concept_id.startswith("ama:") else "fibo")

        linked.append(LinkedEntity(
            mention_id=mention_id,
            canonical_name=canonical_name,
            target_id=concept_id,
            target_label=le.concept_label,
            target_type=le.concept_label,
            target_namespace=namespace,
            status=status,
            confidence=le.confidence,
            ambiguity_candidates=le.ambiguity_candidates,
            notes=notes,
        ))

    hallucination_stats: dict[str, Any] = {
        "reject_count": len(hallucination_rejected),
        "rejected_ids": hallucination_rejected,
        "valid_candidate_count": len(valid_concept_ids),
    }

    return linked, result.usage, hallucination_stats


# ---------------------------------------------------------------------------
# Stage 5: extract_relations (LLM call)
# ---------------------------------------------------------------------------

def extract_relations(
    client: OpenAI,
    *,
    model: str,
    relation_prompt: str,
    mentions: list[EntityMention],
    text: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Call OpenAI to extract semantic relations between entities.

    Returns (relations list, token usage).
    """
    if len(mentions) < 2:
        return [], {}

    mention_descriptions = "\n".join(
        f"- mention_id={m.mention_id} | text=\"{m.text}\" | type={m.entity_type}"
        for m in mentions
    )

    result = structured_completion(
        client,
        model=model,
        system_prompt=relation_prompt,
        user_prompt=(
            f"Extract relations between these entities based on the source text.\n\n"
            f"Entities:\n{mention_descriptions}\n\n"
            f"Source text:\n{text[:3000]}"
        ),
        response_model=RelationExtractionResponse,
    )

    valid_ids = {m.mention_id for m in mentions}
    relations: list[dict[str, Any]] = []
    for rel in result.parsed.relations:
        if rel.source_mention_id in valid_ids and rel.target_mention_id in valid_ids:
            relations.append({
                "source_mention_id": rel.source_mention_id,
                "target_mention_id": rel.target_mention_id,
                "relation_type": rel.relation_type,
                "confidence": rel.confidence,
                "evidence": rel.evidence,
            })
    return relations, result.usage


# ---------------------------------------------------------------------------
# Stage 6: materialize_graph (deterministic, representation-dependent)
# ---------------------------------------------------------------------------

def materialize_graph(
    client: OpenAI,
    *,
    model: str,
    representation: str,
    linked_entities: list[LinkedEntity],
    relations: list[dict[str, Any]],
    text: str,
    example_id: str,
    category: str = "",
    question: str = "",
) -> GraphPreview:
    """Build a deterministic graph preview from linked entities and extracted relations.

    Includes both provenance triples (Document→mentions→Entity) and
    semantic relation triples/edges between entities.
    """
    document_curie = f"ama:{example_id}_document"
    document_node_id = f"doc_{sanitize_node_id(example_id, default='example')}"

    # Build mention_id → local_name lookup for relation materialization
    mention_to_local: dict[str, str] = {}
    for linked in linked_entities:
        local_name = sanitize_node_id(linked.canonical_name or linked.mention_id, default=linked.mention_id)
        mention_to_local[linked.mention_id] = local_name

    if representation == "rdf":
        triples: list[dict[str, str]] = [
            {"subject": document_curie, "predicate": "rdf:type", "object": "ama:Document"},
        ]
        if category:
            triples.append({"subject": document_curie, "predicate": "ama:category", "object": category})
        if question:
            triples.append({"subject": document_curie, "predicate": "ama:questionText", "object": question})

        for linked in linked_entities:
            local_name = mention_to_local[linked.mention_id]
            entity_curie = f"ama:{example_id}_entity_{local_name}"
            triples.append({"subject": document_curie, "predicate": "ama:mentions", "object": entity_curie})
            triples.append({"subject": entity_curie, "predicate": "rdfs:label", "object": linked.canonical_name or linked.mention_id})
            triples.append({"subject": entity_curie, "predicate": "ama:mentionId", "object": linked.mention_id})
            triples.append({"subject": entity_curie, "predicate": "ama:linkStatus", "object": linked.status})
            if linked.confidence is not None:
                triples.append({"subject": entity_curie, "predicate": "ama:confidence", "object": str(linked.confidence)})
            if linked.target_id:
                triples.append({"subject": entity_curie, "predicate": "rdf:type", "object": linked.target_id})
                triples.append({"subject": entity_curie, "predicate": "ama:linkedConcept", "object": linked.target_id})
            if linked.target_label:
                triples.append({"subject": entity_curie, "predicate": "ama:targetLabel", "object": linked.target_label})

        # Semantic relation triples (entity-to-entity with real predicates)
        for rel in relations:
            src_local = mention_to_local.get(rel["source_mention_id"])
            tgt_local = mention_to_local.get(rel["target_mention_id"])
            if src_local and tgt_local:
                src_curie = f"ama:{example_id}_entity_{src_local}"
                tgt_curie = f"ama:{example_id}_entity_{tgt_local}"
                predicate = f"ama:{rel['relation_type']}"
                triples.append({"subject": src_curie, "predicate": predicate, "object": tgt_curie})

        return GraphPreview(triples=triples)

    # --- LPG path ---
    nodes: list[dict[str, Any]] = [
        {
            "id": document_node_id,
            "labels": ["Document"],
            "properties": {
                "example_id": example_id,
                "category": category,
                "question": question,
                "text_length": str(len(text)),
            },
        }
    ]
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = {document_node_id}

    # Build mention_id → node_id lookup for relation edges
    mention_to_node: dict[str, str] = {}

    for linked in linked_entities:
        entity_node_id = f"entity_{sanitize_node_id(linked.canonical_name or linked.mention_id, default=linked.mention_id)}"
        entity_label = sanitize_label(linked.target_label or linked.target_type or "Entity", default="Entity")
        mention_to_node[linked.mention_id] = entity_node_id

        if entity_node_id not in seen_nodes:
            nodes.append(
                {
                    "id": entity_node_id,
                    "labels": ["Entity", entity_label],
                    "properties": {
                        "canonical_name": linked.canonical_name,
                        "mention_id": linked.mention_id,
                        "link_status": linked.status,
                        "confidence": "" if linked.confidence is None else str(linked.confidence),
                        "target_id": linked.target_id,
                        "target_label": linked.target_label,
                    },
                }
            )
            seen_nodes.add(entity_node_id)

        edges.append(
            {
                "source": document_node_id,
                "target": entity_node_id,
                "type": sanitize_relationship_type("MENTIONS"),
                "properties": {"mention_id": linked.mention_id},
            }
        )

        if linked.target_id:
            concept_node_id = f"concept_{sanitize_node_id(linked.target_id, default='concept')}"
            if concept_node_id not in seen_nodes:
                nodes.append(
                    {
                        "id": concept_node_id,
                        "labels": ["OntologyConcept"],
                        "properties": {
                            "concept_id": linked.target_id,
                            "label": linked.target_label,
                            "namespace": linked.target_namespace,
                        },
                    }
                )
                seen_nodes.add(concept_node_id)

            edges.append(
                {
                    "source": entity_node_id,
                    "target": concept_node_id,
                    "type": sanitize_relationship_type("LINKED_TO"),
                    "properties": {"status": linked.status},
                }
            )

    # Semantic relation edges (entity-to-entity with typed relationships)
    for rel in relations:
        src_node = mention_to_node.get(rel["source_mention_id"])
        tgt_node = mention_to_node.get(rel["target_mention_id"])
        if src_node and tgt_node:
            edges.append(
                {
                    "source": src_node,
                    "target": tgt_node,
                    "type": sanitize_relationship_type(rel["relation_type"]),
                    "properties": {
                        "confidence": str(rel.get("confidence", "")),
                        "evidence": rel.get("evidence", ""),
                    },
                }
            )

    return GraphPreview(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Stage 7: compute_basic_metrics (self-consistency, no gold needed)
# ---------------------------------------------------------------------------

def compute_basic_metrics(artifact: ExampleArtifact) -> ExampleMetrics:
    """Compute self-consistency metrics from a single artifact (no gold annotations).

    Metrics computed:
    - null_link_rate: fraction of linked entities with status='null_link'
    - ambiguous_link_rate: fraction with status='ambiguous'
    - materialization_success_rate: 1.0 if graph preview has content, else 0.0
    - duplicate_entity_rate: 1 - (unique / total) canonical names
    - trace_completeness_rate: 1.0 (we got here, so trace is complete)
    """
    metrics = ExampleMetrics()

    linked = artifact.linked_entities
    total = len(linked)
    if total > 0:
        null_count = sum(1 for le in linked if le.status == "null_link")
        ambig_count = sum(1 for le in linked if le.status == "ambiguous")
        metrics.null_link_rate = null_count / total
        metrics.ambiguous_link_rate = ambig_count / total

        canonical_names = [le.canonical_name for le in linked]
        unique_count = len(set(canonical_names))
        metrics.duplicate_entity_rate = 1.0 - (unique_count / total) if total > 0 else 0.0

    gp = artifact.graph_preview
    has_graph = bool(gp.triples or gp.nodes or gp.edges)
    metrics.materialization_success_rate = 1.0 if has_graph else 0.0
    metrics.trace_completeness_rate = 1.0

    return metrics


# ---------------------------------------------------------------------------
# Orchestrator: run_extraction_pipeline
# ---------------------------------------------------------------------------

def run_extraction_pipeline(
    *,
    config: ExperimentConfig,
    example: dict[str, Any],
    representation: str,
    tracer: Any,  # OpikTracer or LocalTraceRecorder
    openai_client: OpenAI,
    model: str,
) -> ExampleArtifact:
    """Run the full extraction pipeline for one example, one representation.

    Each stage is wrapped in a tracer.span() for Opik visibility.
    The pipeline is intentionally linear and explicit:

      1. make_base_artifact    (metadata + ontology + prompts)
      2. join_references       (prepare extraction text)
      3. select_ontology       (none | static | dynamic slice selection)
      4. build_prompts         (selection → guidance pack prompts)
      5. extract_entities      (LLM → entity mentions)
      6. normalize_entities    (deterministic text processing)
      7. link_entities         (LLM → FIBO concept links)
      8. extract_relations     (LLM → semantic relations between entities)
      9. materialize_graph     (deterministic → RDF triples or LPG nodes/edges)
      10. compute_basic_metrics (self-consistency scores)
    """
    pipeline_start = perf_counter()
    category = str(example.get(config.category_field, "unknown"))
    question = str(example.get(config.question_field, ""))

    _span_meta = {"ontology_mode": config.ontology_mode, "representation": representation}

    # --- Stage 1: join references into extraction text ---
    with tracer.span("join_references", inputs={
        "document_field": config.document_field,
    }, metadata=_span_meta) as span_result:
        extraction_text = join_references(example, config.document_field)
        span_result["outputs"] = {
            "text_length": len(extraction_text),
            "text_preview": extraction_text[:500],
        }

    # --- Stage 2: ontology selection ---
    with tracer.span("select_ontology_context", inputs={
        "ontology_mode": config.ontology_mode,
        "category": category,
        "question_preview": question[:300],
        "text_length": len(extraction_text),
        "dynamic_selection_max_packs": config.dynamic_selection_max_packs,
    }, metadata=_span_meta) as span_result:
        ontology_selection = select_ontology_context(
            category=category,
            prompt_template_id=config.prompt_template_id,
            mode=config.ontology_mode,
            question=question,
            references_text=extraction_text,
            openai_client=openai_client,
            model=model,
            max_packs=config.dynamic_selection_max_packs,
        )
        span_result["outputs"] = {
            "selection_mode": ontology_selection.selection_mode,
            "selected_pack_keys": ontology_selection.selected_pack_keys,
            "rejected_pack_keys": ontology_selection.rejected_pack_keys,
            "ontology_modules": ontology_selection.selected_modules,
            "ontology_classes": ontology_selection.selected_classes,
            "selection_reason": ontology_selection.selection_reason,
            "selection_confidence": ontology_selection.selection_confidence,
            "selection_prompt_hash": _prompt_hash(ontology_selection.selection_prompt_preview),
            "selection_prompt_preview": ontology_selection.selection_prompt_preview[:500],
            "notes": ontology_selection.notes[:5],
        }

    # --- Stage 3: build prompt context ---
    with tracer.span("build_prompt_context", inputs={
        "ontology_mode": ontology_selection.selection_mode,
        "selected_pack_keys": ontology_selection.selected_pack_keys,
        "selected_class_count": len(ontology_selection.selected_classes),
    }, metadata=_span_meta) as span_result:
        prompt_context = build_prompt_context(ontology_selection)

        # Register prompts with Opik for version tracking
        _mode = ontology_selection.selection_mode
        prompt_commits = {
            "extraction": _register_opik_prompt(
                f"ama/extraction/{_mode}", prompt_context.extraction_prompt_preview, ontology_mode=_mode,
            ),
            "linking": _register_opik_prompt(
                f"ama/linking/{_mode}", prompt_context.linking_prompt_preview, ontology_mode=_mode,
            ),
            "relation": _register_opik_prompt(
                f"ama/relation/{_mode}", prompt_context.relation_extraction_prompt_preview, ontology_mode=_mode,
            ),
        }

        span_result["outputs"] = {
            "prompt_rule_count": len(prompt_context.rules),
            "prompt_rule_labels": [rule.label for rule in prompt_context.rules[:5]],
            "prompt_hashes": {
                "selection": _prompt_hash(ontology_selection.selection_prompt_preview),
                "extraction": _prompt_hash(prompt_context.extraction_prompt_preview),
                "linking": _prompt_hash(prompt_context.linking_prompt_preview),
                "relation": _prompt_hash(prompt_context.relation_extraction_prompt_preview),
            },
            "prompt_commits": {k: v for k, v in prompt_commits.items() if v},
            "extraction_prompt_preview": prompt_context.extraction_prompt_preview[:500],
            "linking_prompt_preview": prompt_context.linking_prompt_preview[:500],
            "relation_prompt_preview": prompt_context.relation_extraction_prompt_preview[:500],
        }

    # --- Stage 4: base artifact (metadata shell) ---
    with tracer.span("make_base_artifact", inputs={
        "example_id": example.get(config.id_field),
        "representation": representation,
        "category": category,
    }, metadata=_span_meta) as span_result:
        artifact = make_base_artifact(
            config=config,
            example=example,
            representation=representation,
            ontology_selection=ontology_selection,
            prompt_context=prompt_context,
        )
        artifact.raw_input_text = extraction_text
        if not artifact.example_id:
            artifact.example_id = f"auto-{uuid4().hex[:8]}"
            artifact.document_id = artifact.document_id or artifact.example_id
        span_result["outputs"] = {
            "example_id": artifact.example_id,
            "category": artifact.category,
            "ontology_mode": artifact.ontology_selection.selection_mode,
            "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
            "prompt_template_id": artifact.ontology_selection.prompt_template_id,
            "ontology_modules": artifact.ontology_selection.selected_modules,
            "ontology_classes": artifact.ontology_selection.selected_classes,
            "prompt_rule_count": len(artifact.prompt_context.rules),
            "prompt_rule_labels": [rule.label for rule in artifact.prompt_context.rules[:5]],
            "prompt_hashes": {
                "selection": _prompt_hash(artifact.ontology_selection.selection_prompt_preview),
                "extraction": _prompt_hash(artifact.prompt_context.extraction_prompt_preview),
                "linking": _prompt_hash(artifact.prompt_context.linking_prompt_preview),
                "relation": _prompt_hash(artifact.prompt_context.relation_extraction_prompt_preview),
            },
        }

    # --- Stage 5: extract entities (LLM) ---
    with tracer.span("extract_entities", metadata=_span_meta, inputs={
        "model": model,
        "text_length": len(extraction_text),
        "text_preview": extraction_text[:300],
        "ontology_mode": artifact.ontology_selection.selection_mode,
        "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
        "prompt_template_id": artifact.ontology_selection.prompt_template_id,
        "extraction_prompt_hash": _prompt_hash(artifact.prompt_context.extraction_prompt_preview),
        "extraction_prompt_length": len(artifact.prompt_context.extraction_prompt_preview),
        "extraction_prompt_preview": artifact.prompt_context.extraction_prompt_preview[:500],
    }) as span_result:
        mentions, extraction_usage = extract_entities(
            openai_client,
            model=model,
            extraction_prompt=artifact.prompt_context.extraction_prompt_preview,
            text=extraction_text,
        )
        artifact.extracted_entities = mentions
        span_result["outputs"] = {
            "entity_count": len(mentions),
            "entity_types": list(set(m.entity_type for m in mentions)),
            "entity_samples": _mention_sample(mentions),
            "token_usage": extraction_usage,
        }

    # --- Stage 6: normalize entities (deterministic) ---
    with tracer.span("normalize_entities", inputs={
        "mention_count": len(mentions),
    }, metadata=_span_meta) as span_result:
        normalized = normalize_entities(mentions)
        artifact.normalized_entities = normalized
        unique_names = set(n.canonical_name for n in normalized)
        span_result["outputs"] = {
            "normalized_count": len(normalized),
            "unique_canonical_names": len(unique_names),
            "normalized_samples": [
                {
                    "mention_id": item.mention_id,
                    "canonical_name": item.canonical_name,
                    "normalized_type": item.normalized_type,
                }
                for item in normalized[:5]
            ],
        }

    # --- Stage 7: link entities (LLM) ---
    with tracer.span("link_entities", metadata=_span_meta, inputs={
        "model": model,
        "mention_count": len(mentions),
        "text_preview": extraction_text[:300],
        "ontology_mode": artifact.ontology_selection.selection_mode,
        "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
        "prompt_template_id": artifact.ontology_selection.prompt_template_id,
        "linking_prompt_hash": _prompt_hash(artifact.prompt_context.linking_prompt_preview),
        "linking_prompt_length": len(artifact.prompt_context.linking_prompt_preview),
        "linking_prompt_preview": artifact.prompt_context.linking_prompt_preview[:500],
    }) as span_result:
        linked, linking_usage, hallucination_stats = link_entities(
            openai_client,
            model=model,
            linking_prompt=artifact.prompt_context.linking_prompt_preview,
            mentions=mentions,
        )
        artifact.linked_entities = linked
        statuses = [le.status for le in linked]
        span_result["outputs"] = {
            "linked_count": len(linked),
            "status_distribution": {s: statuses.count(s) for s in set(statuses)},
            "linked_samples": _linked_sample(linked),
            "token_usage": linking_usage,
            "hallucination_guard": hallucination_stats,
        }

    # --- Stage 8: extract relations (LLM) ---
    with tracer.span("extract_relations", metadata=_span_meta, inputs={
        "model": model,
        "mention_count": len(mentions),
        "ontology_mode": artifact.ontology_selection.selection_mode,
        "selected_pack_keys": artifact.ontology_selection.selected_pack_keys,
        "prompt_template_id": artifact.ontology_selection.prompt_template_id,
        "relation_prompt_hash": _prompt_hash(artifact.prompt_context.relation_extraction_prompt_preview),
        "relation_prompt_length": len(artifact.prompt_context.relation_extraction_prompt_preview),
        "relation_prompt_preview": artifact.prompt_context.relation_extraction_prompt_preview[:500],
    }) as span_result:
        relations, relation_usage = extract_relations(
            openai_client,
            model=model,
            relation_prompt=artifact.prompt_context.relation_extraction_prompt_preview,
            mentions=mentions,
            text=extraction_text,
        )
        artifact.extracted_relations = relations
        span_result["outputs"] = {
            "relation_count": len(relations),
            "relation_types": list(set(r["relation_type"] for r in relations)),
            "relation_samples": relations[:5],
            "token_usage": relation_usage,
        }

    # --- Stage 9: materialize graph (deterministic) ---
    with tracer.span("materialize_graph", inputs={
        "representation": representation,
        "linked_entity_count": len(linked),
        "relation_count": len(relations),
    }, metadata=_span_meta) as span_result:
        graph_preview = materialize_graph(
            openai_client,
            model=model,
            representation=representation,
            linked_entities=linked,
            relations=relations,
            text=extraction_text,
            example_id=artifact.example_id,
            category=artifact.category,
            question=artifact.question,
        )
        artifact.graph_preview = graph_preview
        span_result["outputs"] = {
            "triple_count": len(graph_preview.triples),
            "node_count": len(graph_preview.nodes),
            "edge_count": len(graph_preview.edges),
            "graph_preview_sample": {
                "triples": graph_preview.triples[:3],
                "nodes": graph_preview.nodes[:3],
                "edges": graph_preview.edges[:3],
            },
        }

    # --- Stage 10: compute basic metrics ---
    with tracer.span("compute_basic_metrics", inputs={
        "representation": representation,
    }, metadata=_span_meta) as span_result:
        metrics = compute_basic_metrics(artifact)
        artifact.metrics = metrics
        artifact.metrics.runtime_ms = int((perf_counter() - pipeline_start) * 1000)
        artifact.status = "completed"
        # Aggregate token usage across all LLM calls
        total_usage = {
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in [extraction_usage, linking_usage, relation_usage]),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in [extraction_usage, linking_usage, relation_usage]),
            "total_tokens": sum(u.get("total_tokens", 0) for u in [extraction_usage, linking_usage, relation_usage]),
        }
        metrics_output = asdict(metrics)
        metrics_output["total_token_usage"] = total_usage
        span_result["outputs"] = metrics_output

    return artifact


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with analysis scripts)
# ---------------------------------------------------------------------------

def placeholder_materialization_preview(representation: str) -> GraphPreview:
    """Generate placeholder graph preview (scaffold-only, no LLM)."""
    if representation == "rdf":
        return GraphPreview(
            triples=[{"subject": "TODO:subject", "predicate": "TODO:predicate", "object": "TODO:object"}]
        )
    return GraphPreview(
        nodes=[
            {"id": "TODO:node-1", "labels": ["Entity"], "properties": {}},
            {"id": "TODO:node-2", "labels": ["Entity"], "properties": {}},
        ],
        edges=[{"source": "TODO:node-1", "target": "TODO:node-2", "type": "RELATED_TO", "properties": {}}],
    )


def artifact_summary(artifact: ExampleArtifact) -> dict[str, Any]:
    return {
        "experiment_name": artifact.experiment_name,
        "experiment_version": artifact.experiment_version,
        "representation": artifact.representation,
        "example_id": artifact.example_id,
        "category": artifact.category,
        "status": artifact.status,
        "entity_count": len(artifact.extracted_entities),
        "linked_entity_count": len(artifact.linked_entities),
        "relation_count": len(artifact.extracted_relations),
        "ontology_selection": asdict(artifact.ontology_selection),
        "prompt_rule_count": len(artifact.prompt_context.rules),
        "metrics": asdict(artifact.metrics),
    }
