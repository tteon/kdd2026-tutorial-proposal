"""
Pydantic response models for OpenAI structured output.

These models define the exact JSON shape the LLM returns at each pipeline stage.
Each model corresponds to one step in the experiment:

  ExtractionResponse   → extract_entities stage
  LinkingResponse      → link_entities stage
  RDFMaterialization   → materialize_graph (RDF path)
  LPGMaterialization   → materialize_graph (LPG path)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage: extract_entities
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """A single entity mention extracted from the input text."""
    text: str = Field(description="The exact mention span from the source text")
    entity_type: str = Field(description="FIBO-aligned entity type, e.g. LegalEntity, Revenue")
    confidence: float = Field(description="Extraction confidence 0.0–1.0")
    evidence: list[str] = Field(default_factory=list, description="Short evidence phrases from the text")
    status: str = Field(default="extracted", description="extracted | uncertain")


class ExtractionResponse(BaseModel):
    """LLM response for the entity extraction stage."""
    entities: list[ExtractedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage: link_entities
# ---------------------------------------------------------------------------

class LinkedEntityResponse(BaseModel):
    """A single entity linked to a FIBO concept."""
    mention_id: str = Field(description="Stable mention identifier from extraction, e.g. m-000")
    mention_text: str = Field(description="Original mention text from extraction")
    concept_id: str = Field(description="Target FIBO concept ID, e.g. fibo:LegalEntity")
    concept_label: str = Field(description="Human-readable concept label")
    confidence: float = Field(description="Linking confidence 0.0–1.0")
    status: str = Field(default="linked", description="linked | ambiguous | null_link")
    ambiguity_candidates: list[str] = Field(default_factory=list, description="Alternative concept IDs considered")


class LinkingResponse(BaseModel):
    """LLM response for the entity linking stage."""
    linked_entities: list[LinkedEntityResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage: extract_relations
# ---------------------------------------------------------------------------

class ExtractedRelation(BaseModel):
    """A single relation between two extracted entities."""
    source_mention_id: str = Field(description="mention_id of the source entity (e.g. m-000)")
    target_mention_id: str = Field(description="mention_id of the target entity (e.g. m-001)")
    relation_type: str = Field(description="Relation type in lowerCamelCase, e.g. hasRevenue, isSubsidiaryOf, isRegulatedBy")
    confidence: float = Field(description="Relation confidence 0.0–1.0")
    evidence: str = Field(default="", description="Brief textual evidence from the source text supporting this relation")


class RelationExtractionResponse(BaseModel):
    """LLM response for the relation extraction stage."""
    relations: list[ExtractedRelation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage: select_ontology_context
# ---------------------------------------------------------------------------

class OntologySelectionResponse(BaseModel):
    """LLM response for dynamic ontology slice selection."""
    selected_pack_keys: list[str] = Field(
        default_factory=list,
        description="One to three ontology pack keys chosen from the provided candidate list.",
    )
    rejected_pack_keys: list[str] = Field(
        default_factory=list,
        description="Candidate pack keys considered but not selected.",
    )
    selection_reason: str = Field(
        default="",
        description="Short explanation of why the selected packs fit the question and evidence.",
    )
    selection_confidence: float = Field(
        default=0.0,
        description="Selection confidence from 0.0 to 1.0.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Optional notes about ambiguity, fallback, or pack coverage gaps.",
    )


# ---------------------------------------------------------------------------
# Stage: materialize_graph (RDF path)
# ---------------------------------------------------------------------------

class RDFTriple(BaseModel):
    """A single subject-predicate-object triple."""
    subject: str = Field(description="Subject URI or canonical name")
    predicate: str = Field(description="Predicate URI or relation name")
    object: str = Field(description="Object URI, canonical name, or literal value")


class RDFMaterialization(BaseModel):
    """LLM response for RDF graph materialization."""
    triples: list[RDFTriple] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage: materialize_graph (LPG path)
# ---------------------------------------------------------------------------

class KeyValue(BaseModel):
    """A single key-value property pair (used instead of dict for OpenAI schema compatibility)."""
    key: str = Field(description="Property name")
    value: str = Field(description="Property value")


class LPGNode(BaseModel):
    """A labeled property graph node."""
    id: str = Field(description="Unique node identifier")
    labels: list[str] = Field(default_factory=list, description="Node labels, e.g. ['LegalEntity']")
    node_properties: list[KeyValue] = Field(default_factory=list, description="Node property key-value pairs")


class LPGEdge(BaseModel):
    """A labeled property graph edge."""
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: str = Field(description="Relationship type, e.g. HAS_REVENUE")
    edge_properties: list[KeyValue] = Field(default_factory=list, description="Edge property key-value pairs")


class LPGMaterialization(BaseModel):
    """LLM response for LPG graph materialization."""
    nodes: list[LPGNode] = Field(default_factory=list)
    edges: list[LPGEdge] = Field(default_factory=list)
