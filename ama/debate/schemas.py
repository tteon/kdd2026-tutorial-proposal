"""
Agent I/O contracts for the RDF vs LPG debate pool.

These Pydantic models define the structured output format for each agent,
following the contracts specified in EXPERIMENT_SETUP.md §Search Agent JSON Contract.

Hierarchy:
  RDFAgentOutput / LPGAgentOutput  — independent generation
  CritiqueOutput                   — debate critique round
  SynthesisOutput                  — final synthesized answer
  ComparisonResult                 — output comparison analysis
  HandoffLog                       — debate decision audit trail
  DebateResult                     — full pipeline result
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Claims and evidence (shared building blocks)
# ---------------------------------------------------------------------------

class RDFClaim(BaseModel):
    """A claim from the RDF agent, grounded in typed triples."""
    claim_id: str = Field(description="Stable claim identifier, e.g. rc-001")
    text: str = Field(description="Natural-language claim statement")
    linked_entities: list[str] = Field(default_factory=list, description="FIBO concept IDs supporting this claim")
    typed_relations: list[str] = Field(default_factory=list, description="Predicate URIs used as evidence")
    confidence: float = Field(default=0.0, description="Claim confidence 0.0–1.0")
    status: str = Field(default="supported", description="supported | partial | unsupported")


class LPGClaim(BaseModel):
    """A claim from the LPG agent, grounded in nodes/edges/properties."""
    claim_id: str = Field(description="Stable claim identifier, e.g. lc-001")
    text: str = Field(description="Natural-language claim statement")
    node_ids: list[str] = Field(default_factory=list, description="Node IDs referenced")
    edge_types: list[str] = Field(default_factory=list, description="Edge types traversed")
    property_paths: list[str] = Field(default_factory=list, description="Property paths used")
    confidence: float = Field(default=0.0, description="Claim confidence 0.0–1.0")
    status: str = Field(default="supported", description="supported | partial | unsupported")


class EvidenceItem(BaseModel):
    """A piece of graph evidence supporting one or more claims."""
    evidence_id: str = Field(description="Stable evidence identifier")
    source_type: str = Field(description="triple | node | edge | property | document_chunk")
    source_ref: str = Field(description="URI, node ID, or document reference")
    content: str = Field(description="Human-readable evidence content")
    supports_claim_ids: list[str] = Field(default_factory=list, description="Which claims this supports")


class ToolTraceDigest(BaseModel):
    """Compact summary of one tool use for later synthesis and trace inspection."""
    tool_name: str = Field(description="Tool name, e.g. search_rdf_entities")
    query_intent: str = Field(default="", description="Why the tool was used")
    query_text: str = Field(default="", description="Query text or parameter summary")
    results_summary: str = Field(default="", description="Compact summary of what the tool returned")
    analysis: str = Field(default="", description="How the agent interpreted the tool result")


class RetrievalDiagnostics(BaseModel):
    """Structured retrieval diagnostics for Search-A style graph access."""
    question_type: str = Field(default="", description="entity-centric | relation-centric | value-metric | evidence-heavy")
    retrieval_plan: list[str] = Field(default_factory=list, description="Ordered retrieval steps attempted")
    anchor_terms: list[str] = Field(default_factory=list, description="Anchor entities/tickers/concepts extracted from the question")
    template_used: list[str] = Field(default_factory=list, description="Retrieval templates used")
    lexical_fallback_used: bool = Field(default=False, description="Whether lexical fallback was used")
    freeform_query_used: bool = Field(default=False, description="Whether free-form query was used")
    invalid_schema_assumptions: list[str] = Field(default_factory=list, description="Invented labels/edges/properties or schema mismatches encountered")


# ---------------------------------------------------------------------------
# Agent outputs — independent generation phase
# ---------------------------------------------------------------------------

class RDFAgentOutput(BaseModel):
    """Structured output from the RDF agent (EXPERIMENT_SETUP.md contract)."""
    representation: str = "rdf"
    answer_draft: str = Field(description="Draft answer grounded in RDF graph evidence")
    claims: list[RDFClaim] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    tool_trace_digest: list[ToolTraceDigest] = Field(default_factory=list)
    question_type: str = Field(default="", description="Search-A question type selected before retrieval")
    retrieval_support_level: str = Field(default="no_support", description="direct_support | indirect_support | no_support")
    retrieval_diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    confidence: float = Field(default=0.0, description="Overall answer confidence 0.0–1.0")
    uncertainty_notes: list[str] = Field(default_factory=list, description="What the agent is unsure about")
    missing_evidence: list[str] = Field(default_factory=list, description="Evidence the agent looked for but didn't find")


class LPGAgentOutput(BaseModel):
    """Structured output from the LPG agent (EXPERIMENT_SETUP.md contract)."""
    representation: str = "lpg"
    answer_draft: str = Field(description="Draft answer grounded in LPG graph evidence")
    claims: list[LPGClaim] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    tool_trace_digest: list[ToolTraceDigest] = Field(default_factory=list)
    question_type: str = Field(default="", description="Search-A question type selected before retrieval")
    retrieval_support_level: str = Field(default="no_support", description="direct_support | indirect_support | no_support")
    retrieval_diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    confidence: float = Field(default=0.0, description="Overall answer confidence 0.0–1.0")
    uncertainty_notes: list[str] = Field(default_factory=list, description="What the agent is unsure about")
    missing_evidence: list[str] = Field(default_factory=list, description="Evidence the agent looked for but didn't find")


# ---------------------------------------------------------------------------
# Critique output — debate round
# ---------------------------------------------------------------------------

class CritiqueOutput(BaseModel):
    """Output from a debate critique round.

    Each agent receives the other side's claims and evidence,
    then produces critiques, concessions, and a revised answer.
    """
    critiqued_claims: list[dict] = Field(
        default_factory=list,
        description="List of {claim_id, critique, verdict: agree|disagree|partial}",
    )
    revised_answer: str = Field(default="", description="Revised answer after considering critique")
    revised_confidence: float = Field(default=0.0, description="Updated confidence after critique")
    conceded_points: list[str] = Field(default_factory=list, description="Points the agent concedes to the other side")
    maintained_points: list[str] = Field(default_factory=list, description="Points the agent stands by")


# ---------------------------------------------------------------------------
# Synthesis output — final answer
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    """Structured output from the answer synthesis agent (EXPERIMENT_SETUP.md contract)."""
    final_answer: str = Field(description="Synthesized final answer")
    rdf_support_summary: list[str] = Field(default_factory=list, description="What RDF evidence supports")
    lpg_support_summary: list[str] = Field(default_factory=list, description="What LPG evidence supports")
    agreement_points: list[str] = Field(default_factory=list, description="Where RDF and LPG agree")
    disagreement_points: list[str] = Field(default_factory=list, description="Where they disagree")
    unresolved_points: list[str] = Field(default_factory=list, description="What remains unclear")
    final_confidence: float = Field(default=0.0, description="Synthesis confidence 0.0–1.0")
    resolution_mode: str = Field(
        default="direct_synthesis",
        description="direct_synthesis | self_reflection | debate | judge",
    )
    selected_supporting_representation: str = Field(
        default="hybrid",
        description="rdf | lpg | hybrid — which representation primarily supports the answer",
    )


# ---------------------------------------------------------------------------
# Comparison and debate orchestration
# ---------------------------------------------------------------------------

class ComparisonResult(BaseModel):
    """Result of comparing RDF and LPG agent outputs.

    Used by the debate pool to decide which strategy to use.
    """
    answers_agree: bool = Field(default=False, description="Do the two answers materially agree?")
    confidence_gap: float = Field(default=0.0, description="abs(rdf_confidence - lpg_confidence)")
    evidence_overlap_ratio: float = Field(default=0.0, description="Fraction of shared evidence references")
    rdf_null_or_ambiguous: bool = Field(default=False, description="RDF agent produced a null/weak answer")
    lpg_null_or_ambiguous: bool = Field(default=False, description="LPG agent produced a null/weak answer")
    strategy_rationale: str = Field(default="", description="Why the router policy recommended this strategy")
    decision_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Thresholds used by the deterministic router policy",
    )
    recommended_strategy: str = Field(
        default="direct_synthesis",
        description="direct_synthesis | self_reflection | debate | judge",
    )


class HandoffLog(BaseModel):
    """Audit trail for debate handoff decisions (logged to Opik trace).

    These fields answer: why did the system choose this resolution strategy?
    """
    initial_answer_agreement: bool = False
    initial_confidence_gap: float = 0.0
    initial_evidence_overlap: float = 0.0
    rdf_null_or_ambiguous: bool = False
    lpg_null_or_ambiguous: bool = False
    self_reflection_trigger: bool = False
    debate_trigger: bool = False
    judge_trigger: bool = False
    strategy_rationale: str = ""
    decision_thresholds: dict[str, float] = Field(default_factory=dict)
    final_resolution_mode: str = "direct_synthesis"


class AgentTraceRecord(BaseModel):
    """Cross-system trace linkage for one agent run."""
    stage_name: str = Field(description="Pipeline stage, e.g. run_rdf_agent")
    agent_name: str = Field(description="OpenAI agent name")
    openai_trace_id: str = Field(default="", description="OpenAI Agents SDK trace ID")
    openai_workflow_name: str = Field(default="", description="OpenAI workflow name")
    openai_group_id: str = Field(default="", description="OpenAI trace group ID")
    openai_response_id: str = Field(default="", description="Last OpenAI response ID for the run")
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Trace metadata recorded on the OpenAI side",
    )
    tracing_enabled: bool = Field(default=False, description="Whether an OpenAI trace ID was captured")


class DebateTraceLinkage(BaseModel):
    """How the debate run is linked across Opik and OpenAI tracing systems."""
    opik_trace_id: str = Field(default="", description="Top-level Opik trace ID")
    opik_backend: str = Field(default="", description="opik_cloud or local_json")
    opik_graph_definition: str = Field(default="", description="Mermaid graph stored in Opik metadata")
    openai_traces: dict[str, AgentTraceRecord] = Field(
        default_factory=dict,
        description="Per-stage OpenAI trace linkage keyed by stage name",
    )


class DebateResult(BaseModel):
    """Complete result of one debate pool run.

    Contains all intermediate outputs for inspection and tracing.
    """
    question: str
    representation_condition: str = Field(
        default="representation_preserving_joint",
        description="rdf_only | lpg_only | representation_preserving_joint",
    )
    rdf_output: RDFAgentOutput
    lpg_output: LPGAgentOutput
    comparison: ComparisonResult
    handoff_log: HandoffLog
    synthesis: SynthesisOutput
    rdf_critique: Optional[CritiqueOutput] = None
    lpg_critique: Optional[CritiqueOutput] = None
    trace_id: str = ""
    trace_linkage: DebateTraceLinkage = Field(default_factory=DebateTraceLinkage)
