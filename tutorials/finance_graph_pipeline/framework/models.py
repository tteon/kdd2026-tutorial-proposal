from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    section_type: str
    text: str


@dataclass
class Question:
    question_id: str
    question: str
    query_template: str
    target_profile: str
    ground_truth_answer: str


@dataclass
class FiboProfileDecision:
    selected_profile: str
    candidate_profiles: list[str]
    selection_confidence: float
    mapping_policy: str
    ontology_rationale: str


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    confidence: float
    source_doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    source_name: str
    relation_type: str
    target_name: str
    confidence: float
    source_doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    node_id: str
    name: str
    entity_type: str
    profile: str
    aliases: set[str] = field(default_factory=set)
    source_doc_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    edge_id: str
    source_node_id: str
    relation_type: str
    target_node_id: str
    profile: str
    source_doc_id: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityIssue:
    issue_type: str
    severity: str
    message: str
    object_type: str
    object_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuerySupportRecord:
    question_id: str
    query_template: str
    target_profile: str
    support_score: float
    answerable: bool
    supporting_edge_ids: list[str]
    missing_requirements: list[str]


@dataclass
class QuestionRoutingDecision:
    selected_profile: str
    query_template: str
    evidence_strategy: str
    rationale: str


@dataclass
class AnswerResult:
    question_id: str
    answer: str
    confidence: float
    selected_edge_ids: list[str]
    routing: QuestionRoutingDecision
    quality_notes: list[str]


@dataclass
class GraphState:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    profile_decisions: dict[str, FiboProfileDecision] = field(default_factory=dict)
    quality_issues: list[QualityIssue] = field(default_factory=list)
    query_support: dict[str, QuerySupportRecord] = field(default_factory=dict)
