from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class OntologySelection:
    selection_mode: str = "static"
    selected_pack_keys: list[str] = field(default_factory=list)
    selected_modules: list[str] = field(default_factory=list)
    selected_classes: list[str] = field(default_factory=list)
    rejected_pack_keys: list[str] = field(default_factory=list)
    selection_reason: str = ""
    selection_confidence: Optional[float] = None
    prompt_template_id: str = ""
    selection_prompt_preview: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class EntityMention:
    mention_id: str
    text: str
    entity_type: str = ""
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    source_text_id: str = ""
    confidence: Optional[float] = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class NormalizedEntity:
    mention_id: str
    canonical_name: str
    normalized_type: str = ""
    normalization_method: str = ""
    confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class LinkedEntity:
    mention_id: str
    canonical_name: str
    target_id: str = ""
    target_label: str = ""
    target_type: str = ""
    target_namespace: str = "fibo"
    status: str = "unlinked"
    confidence: Optional[float] = None
    ambiguity_candidates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PromptRule:
    concept_id: str
    label: str
    definition: str = ""
    synonyms: list[str] = field(default_factory=list)
    semantic_features: list[str] = field(default_factory=list)
    soft_signals: list[str] = field(default_factory=list)
    extraction_hint: str = ""
    linking_hint: str = ""


@dataclass
class PromptContext:
    rules: list[PromptRule] = field(default_factory=list)
    extraction_prompt_preview: str = ""
    linking_prompt_preview: str = ""
    relation_extraction_prompt_preview: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class GraphPreview:
    triples: list[dict[str, str]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExampleMetrics:
    entity_extraction_precision: Optional[float] = None
    entity_extraction_recall: Optional[float] = None
    entity_extraction_f1: Optional[float] = None
    entity_linking_accuracy: Optional[float] = None
    canonicalization_consistency_rate: Optional[float] = None
    materialization_success_rate: Optional[float] = None
    unsupported_entity_rate: Optional[float] = None
    ambiguous_link_rate: Optional[float] = None
    null_link_rate: Optional[float] = None
    duplicate_entity_rate: Optional[float] = None
    schema_or_type_conformance_rate: Optional[float] = None
    trace_completeness_rate: Optional[float] = None
    fallback_error_rate: Optional[float] = None
    runtime_ms: Optional[int] = None


@dataclass
class ExampleArtifact:
    experiment_name: str
    experiment_version: str
    representation: str
    example_id: str
    document_id: str
    question_id: str = ""
    category: str = ""
    profile_id: str = ""
    source_dataset: str = ""
    source_split: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    question: str = ""
    raw_input_text: str = ""
    chunk_reference: str = ""
    ontology_selection: OntologySelection = field(default_factory=OntologySelection)
    prompt_context: PromptContext = field(default_factory=PromptContext)
    extracted_entities: list[EntityMention] = field(default_factory=list)
    extracted_relations: list[dict[str, Any]] = field(default_factory=list)
    normalized_entities: list[NormalizedEntity] = field(default_factory=list)
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    unresolved_entities: list[dict[str, Any]] = field(default_factory=list)
    graph_preview: GraphPreview = field(default_factory=GraphPreview)
    rdf_export: dict[str, Any] = field(default_factory=dict)
    metrics: ExampleMetrics = field(default_factory=ExampleMetrics)
    trace_id: str = ""
    trace_backend: str = ""
    status: str = "pending"
    error_notes: list[str] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    experiment_name: str
    experiment_version: str
    dataset_id: str
    dataset_split: str
    sample_size: int
    random_seed: int = 7
    category_field: str = "category"
    question_field: str = "text"
    document_field: str = "references"
    answer_field: str = "answer"
    id_field: str = "_id"
    ontology_mode: str = "all_packs"
    dynamic_selection_max_packs: int = 3
    prompt_template_id: str = "finder_fibo_v1"
    link_target_namespace: str = "fibo"
    opik_project_name: str = "ama_rdf_lpg_observability"
    output_dir: str = "artifacts"
    profile_selection_enabled: bool = True


@dataclass
class DivergenceSummary:
    shared_canonical_names: list[str] = field(default_factory=list)
    rdf_only_canonical_names: list[str] = field(default_factory=list)
    lpg_only_canonical_names: list[str] = field(default_factory=list)
    differing_link_targets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExampleChainView:
    example_id: str
    category: str = ""
    question: str = ""
    references_preview: str = ""
    ontology: dict[str, Any] = field(default_factory=dict)
    rdf_view: dict[str, Any] = field(default_factory=dict)
    lpg_view: dict[str, Any] = field(default_factory=dict)
    divergence: DivergenceSummary = field(default_factory=DivergenceSummary)
    debate_view: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
