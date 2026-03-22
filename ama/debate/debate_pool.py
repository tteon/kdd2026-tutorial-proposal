"""
Debate Pool — orchestrates the RDF vs LPG agent debate pipeline.

Pipeline (each stage is a separate traced method):

  1. generate_independently  → both agents query their graphs
  2. compare_outputs         → analyze agreement/disagreement
  3. decide_strategy         → deterministic router policy chooses direct_synthesis | self_reflection | debate | judge
  4. execute_strategy        → run the chosen resolution path
  5. synthesize              → synthesis agent produces final answer

Strategy decision (from EXPERIMENT_SETUP.md §Debate Handoff Decision Policy):

  direct_synthesis    — answers agree, both have acceptable evidence
  self_reflection     — one agent is weak but no clear competing claim
  debate              — answers disagree, evidence points to different interpretations
  judge               — persistent disagreement after critique round

Important:
  The router is currently a deterministic policy, not a separate LLM agent.
  This keeps handoff logic attributable and limits prompt variance during the current observability phase.

Usage:
    from debate.debate_pool import DebatePool

    pool = DebatePool(model="gpt-4o")
    result = pool.run("What is Apple's governance structure?", experiment_name="demo")
    print(result.synthesis.final_answer)
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from agents import Agent, AgentOutputSchema, RunConfig, Runner  # OpenAI Agents SDK

from debate.graph_tools import (
    close_connections,
    get_active_scope,
    get_database_bindings,
    get_lpg_schema_card,
    get_rdf_schema_card,
    init_connections,
)
from debate.lpg_agent import make_lpg_agent
from debate.rdf_agent import make_rdf_agent
from debate.schemas import (
    AgentTraceRecord,
    ComparisonResult,
    CritiqueOutput,
    DebateResult,
    DebateTraceLinkage,
    HandoffLog,
    LPGAgentOutput,
    RDFAgentOutput,
    SynthesisOutput,
)
from debate.synthesis_agent import make_synthesis_agent


# Confidence threshold: below this, the agent is considered "null or ambiguous"
CONFIDENCE_THRESHOLD = 0.3

# Agreement threshold: if both answers are this similar in meaning, treat as agreeing
# (checked via a lightweight LLM call in compare_outputs)
AGREEMENT_CONFIDENCE_GAP = 0.3


class DebateExecutionError(RuntimeError):
    """Stage-aware wrapper so debate failures stay attributable in artifacts."""

    def __init__(
        self,
        *,
        stage_name: str,
        original_exception: Exception,
        experiment_name: str,
        target_databases: dict[str, str] | None = None,
        active_scope: dict[str, str] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.original_exception = original_exception
        self.exception_type = type(original_exception).__name__
        self.experiment_name = experiment_name
        self.target_databases = dict(target_databases or {})
        self.active_scope = dict(active_scope or {})
        super().__init__(
            f"Debate stage {stage_name} failed ({self.exception_type}): {original_exception}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "exception_type": self.exception_type,
            "message": str(self.original_exception),
            "experiment_name": self.experiment_name,
            "target_databases": self.target_databases,
            "active_scope": self.active_scope,
        }


def _truncate_text(value: str, *, limit: int = 240) -> str:
    """Keep trace payloads readable without dropping the whole answer."""
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _tool_digest_summary(tool_digests: list[Any], *, limit: int = 3) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for item in tool_digests[:limit]:
        summary.append(
            {
                "tool_name": getattr(item, "tool_name", ""),
                "query_intent": getattr(item, "query_intent", ""),
                "results_summary": _truncate_text(getattr(item, "results_summary", ""), limit=120),
                "analysis": _truncate_text(getattr(item, "analysis", ""), limit=120),
            }
        )
    return summary


def _evidence_key_set(evidence_items: list[Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for item in evidence_items:
        source_type = getattr(item, "source_type", "")
        source_ref = getattr(item, "source_ref", "")
        if source_type or source_ref:
            keys.add((source_type, source_ref))
    return keys


def _debate_run_kind(experiment_name: str) -> str:
    lower = experiment_name.lower()
    if "category" in lower:
        return "category_pilot"
    if "pilot" in lower:
        return "pilot"
    if "smoke" in lower or "check" in lower:
        return "smoke"
    return "run"


def _debate_trace_name(*, experiment_name: str, question: str) -> str:
    run_kind = _debate_run_kind(experiment_name)
    return f"debate/{run_kind}/{experiment_name}"


def _classify_question_type(question: str) -> str:
    lowered = question.lower()
    if any(token in lowered for token in ["yoy", "year-over-year", "margin", "revenue", "eps", "price", "volume", "2021", "2022", "2023", "2024"]):
        return "value-metric"
    if any(token in lowered for token in ["impact", "implications", "outlook", "strategy", "positioning", "risk profile", "profitability", "valuation"]):
        return "evidence-heavy"
    if any(token in lowered for token in [" vs ", "affect", "influence", "driver", "relationship", "between"]):
        return "relation-centric"
    return "entity-centric"


def _extract_anchor_terms(question: str) -> list[str]:
    terms: list[str] = []
    for ticker in re.findall(r"\(([A-Z][A-Z0-9.\-]{1,6})\)", question):
        if ticker not in terms:
            terms.append(ticker)
    for value in re.findall(r"\b[A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+){0,3}\b", question):
        cleaned = value.strip(" .,:;()")
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return terms[:8]


def _search_a_plan(question_type: str) -> list[str]:
    base = ["anchor_lookup", "local_context_expansion"]
    if question_type == "relation-centric":
        return base + ["relation_probe"]
    if question_type == "value-metric":
        return base + ["metric_probe"]
    if question_type == "evidence-heavy":
        return base + ["document_evidence_expansion", "relation_probe"]
    return base + ["document_evidence_expansion"]


def _build_agent_prompt(
    *,
    question: str,
    question_type: str,
    anchor_terms: list[str],
    retrieval_plan: list[str],
    schema_card: dict[str, Any],
    representation: str,
) -> str:
    schema_lines: list[str] = []
    if representation == "rdf":
        schema_lines.append(f"Observed labels: {', '.join(schema_card.get('labels', [])) or 'none'}")
        schema_lines.append(f"Observed predicates: {', '.join(schema_card.get('predicates', [])) or 'none'}")
    else:
        schema_lines.append(f"Observed labels: {', '.join(schema_card.get('labels', [])) or 'none'}")
        schema_lines.append(f"Observed relationship types: {', '.join(schema_card.get('relationship_types', [])) or 'none'}")
    schema_lines.append(f"Forbidden assumptions: {', '.join(schema_card.get('forbidden_assumptions', []))}")
    return (
        f"Question: {question}\n\n"
        f"Search-A scaffold:\n"
        f"- question_type: {question_type}\n"
        f"- anchor_terms: {anchor_terms}\n"
        f"- retrieval_plan: {retrieval_plan}\n"
        f"- schema_card:\n  - " + "\n  - ".join(schema_lines) + "\n\n"
        f"Execution constraints:\n"
        f"- start with anchor lookup\n"
        f"- inspect local context before free-form query\n"
        f"- use free-form query only if the template path is insufficient\n"
        f"- report retrieval_support_level as direct_support, indirect_support, or no_support\n"
    )


def _coerce_agent_output(agent_name: str, output: Any) -> Any:
    """Normalize occasional wrapped/stringified agent outputs from the SDK."""
    expected_model: Any | None = None
    if "RDF" in agent_name and "Critique" not in agent_name:
        expected_model = RDFAgentOutput
    elif "LPG" in agent_name and "Critique" not in agent_name:
        expected_model = LPGAgentOutput
    elif "Synthesis" in agent_name:
        expected_model = SynthesisOutput
    elif "Critique" in agent_name:
        expected_model = CritiqueOutput

    if expected_model is None or isinstance(output, expected_model):
        return output

    candidate = output
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return output

    if isinstance(candidate, dict) and isinstance(candidate.get("representation"), str):
        representation_payload = candidate.get("representation", "").strip()
        if representation_payload.startswith("{") and representation_payload.endswith("}"):
            try:
                candidate = json.loads(representation_payload)
            except json.JSONDecodeError:
                pass

    if (
        isinstance(candidate, dict)
        and isinstance(candidate.get("description"), str)
        and isinstance(candidate.get("properties"), dict)
    ):
        candidate = candidate["properties"]

    if isinstance(candidate, dict):
        try:
            return expected_model.model_validate(candidate)
        except Exception:
            return output
    return output


def _infer_support_level(output: Any) -> str:
    if getattr(output, "evidence_items", None):
        return "direct_support"
    if getattr(output, "claims", None) or getattr(output, "tool_trace_digest", None):
        return "indirect_support"
    return "no_support"


def _backfill_retrieval_fields(
    output: Any,
    *,
    question_type: str,
    anchor_terms: list[str],
    retrieval_plan: list[str],
) -> Any:
    if not getattr(output, "question_type", ""):
        output.question_type = question_type
    if not getattr(output, "retrieval_support_level", "") or output.retrieval_support_level == "no_support":
        output.retrieval_support_level = _infer_support_level(output)
    diagnostics = getattr(output, "retrieval_diagnostics", None)
    if diagnostics is not None:
        if not diagnostics.question_type:
            diagnostics.question_type = question_type
        if not diagnostics.retrieval_plan:
            diagnostics.retrieval_plan = list(retrieval_plan)
        if not diagnostics.anchor_terms:
            diagnostics.anchor_terms = list(anchor_terms)
        if not diagnostics.template_used:
            diagnostics.template_used = [step for step in retrieval_plan if step not in ("anchor_lookup", "local_context_expansion")]
        if not diagnostics.invalid_schema_assumptions:
            bad_tokens: list[str] = []
            for item in getattr(output, "tool_trace_digest", []):
                analysis = f"{getattr(item, 'results_summary', '')} {getattr(item, 'analysis', '')}"
                for token in ["RELATED_TO", "HAS_FINANCIAL", "PAYS_DIVIDEND", "FinancialMetric", "value"]:
                    if token.lower() in analysis.lower() and token not in bad_tokens:
                        bad_tokens.append(token)
            diagnostics.invalid_schema_assumptions = bad_tokens
        diagnostics.freeform_query_used = any(
            "query_" in getattr(item, "tool_name", "") and "cypher" in getattr(item, "tool_name", "")
            for item in getattr(output, "tool_trace_digest", [])
        )
    return output


class DebatePool:
    """Orchestrates the RDF vs LPG debate pipeline.

    Each method corresponds to one pipeline stage, making the flow
    explicit and traceable. All intermediate outputs are preserved
    in the DebateResult for inspection.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        tracer: Any = None,
        retrieval_variant: str = "search_a_v1",
    ) -> None:
        self.model = model
        self.tracer = tracer
        self.retrieval_variant = retrieval_variant

        allow_freeform_queries = retrieval_variant not in {"search_a_v3", "search_a_v4", "search_a_v5"}
        schema_guard_level = "strict" if retrieval_variant in {"search_a_v4", "search_a_v5"} else "normal"

        self.rdf_agent = make_rdf_agent(
            model=model,
            allow_freeform_queries=allow_freeform_queries,
            schema_guard_level=schema_guard_level,
        )
        self.lpg_agent = make_lpg_agent(
            model=model,
            allow_freeform_queries=allow_freeform_queries,
            schema_guard_level=schema_guard_level,
        )
        self.synthesis_agent = make_synthesis_agent(model=model)

        # Critique agents are variants with different instructions
        self._rdf_critique_agent = self._make_critique_agent("RDF", model)
        self._lpg_critique_agent = self._make_critique_agent("LPG", model)

    def _build_agent_graph_definition(self) -> str:
        """Mermaid graph for Opik's manual agent-graph logging."""
        return "\n".join([
            "flowchart TD",
            '    Q["Question"] --> RDF["RDFAgent"]',
            '    Q["Question"] --> LPG["LPGAgent"]',
            '    RDF["RDFAgent"] --> CMP["compare_outputs"]',
            '    LPG["LPGAgent"] --> CMP["compare_outputs"]',
            '    CMP["compare_outputs"] --> DEC["decide_strategy"]',
            '    DEC["decide_strategy"] -->|direct_synthesis| SYN["SynthesisAgent"]',
            '    DEC["decide_strategy"] -->|self_reflection| RC["RDF Critique Agent"]',
            '    DEC["decide_strategy"] -->|self_reflection| LC["LPG Critique Agent"]',
            '    DEC["decide_strategy"] -->|debate| RC["RDF Critique Agent"]',
            '    DEC["decide_strategy"] -->|debate| LC["LPG Critique Agent"]',
            '    DEC["decide_strategy"] -->|judge| RC["RDF Critique Agent"]',
            '    DEC["decide_strategy"] -->|judge| LC["LPG Critique Agent"]',
            '    RC["RDF Critique Agent"] --> SYN["SynthesisAgent"]',
            '    LC["LPG Critique Agent"] --> SYN["SynthesisAgent"]',
            '    SYN["SynthesisAgent"] --> OUT["Final Answer"]',
        ])

    def _make_run_config(
        self,
        *,
        stage_name: str,
        agent_name: str,
        experiment_name: str,
        question: str,
        opik_trace_id: str,
        strategy: str = "",
        other_representation: str = "",
        target_database: str = "",
    ) -> RunConfig:
        """Build a RunConfig so OpenAI-side traces can be linked back to Opik."""
        trace_metadata = {
            "experiment_name": experiment_name,
            "opik_trace_id": opik_trace_id,
            "debate_stage": stage_name,
            "agent_name": agent_name,
            "question_length": str(len(question)),
        }
        if strategy:
            trace_metadata["resolution_mode"] = strategy
        if other_representation:
            trace_metadata["other_representation"] = _truncate_text(other_representation, limit=480)
        if target_database:
            trace_metadata["target_database"] = target_database

        group_id = opik_trace_id or experiment_name or "debate"
        return RunConfig(
            workflow_name=stage_name,
            group_id=group_id,
            trace_metadata=trace_metadata,
        )

    def _extract_trace_record(
        self,
        *,
        result: Any,
        stage_name: str,
        agent_name: str,
    ) -> AgentTraceRecord:
        """Extract trace linkage details from an OpenAI Agents SDK run result."""
        trace_obj = getattr(result, "trace", None)
        trace_state = getattr(result, "_trace_state", None)

        openai_trace_id = (
            getattr(trace_obj, "trace_id", None)
            or getattr(trace_obj, "id", None)
            or getattr(trace_state, "trace_id", None)
            or ""
        )
        workflow_name = (
            getattr(trace_state, "workflow_name", None)
            or getattr(trace_obj, "workflow_name", None)
            or ""
        )
        group_id = (
            getattr(trace_state, "group_id", None)
            or getattr(trace_obj, "group_id", None)
            or ""
        )
        trace_metadata = getattr(trace_state, "metadata", None) or {}
        last_response_id = getattr(result, "last_response_id", None) or ""

        return AgentTraceRecord(
            stage_name=stage_name,
            agent_name=agent_name,
            openai_trace_id=str(openai_trace_id),
            openai_workflow_name=str(workflow_name),
            openai_group_id=str(group_id),
            openai_response_id=str(last_response_id),
            trace_metadata=trace_metadata,
            tracing_enabled=bool(openai_trace_id),
        )

    async def _run_agent_with_trace(
        self,
        agent: Agent,
        prompt: str,
        *,
        stage_name: str,
        experiment_name: str,
        opik_trace_id: str,
        strategy: str = "",
        other_representation: str = "",
        target_database: str = "",
    ) -> tuple[Any, AgentTraceRecord]:
        """Run one agent and return both its output and OpenAI trace linkage."""
        run_config = self._make_run_config(
            stage_name=stage_name,
            agent_name=agent.name,
            experiment_name=experiment_name,
            question=prompt,
            opik_trace_id=opik_trace_id,
            strategy=strategy,
            other_representation=other_representation,
            target_database=target_database,
        )
        result = await Runner.run(agent, prompt, run_config=run_config)
        final_output = _coerce_agent_output(agent.name, result.final_output)
        trace_record = self._extract_trace_record(
            result=result,
            stage_name=stage_name,
            agent_name=agent.name,
        )
        return final_output, trace_record

    def _make_critique_agent(self, representation: str, model: str) -> Agent:
        """Create a critique variant of an agent for the debate round."""
        other = "LPG" if representation == "RDF" else "RDF"
        return Agent(
            name=f"{representation} Critique Agent",
            instructions=f"""\
You are the {representation} agent in a debate with the {other} agent.

You have been given the {other} agent's claims and evidence.
Your task is to critique their claims using your own graph knowledge.

For each claim from the other side:
- If you agree: concede the point
- If you disagree: explain why, citing your own evidence
- If partially agree: note what's correct and what's wrong

Then produce a revised answer that accounts for the other side's valid points.
Be honest — if the other side found evidence you missed, acknowledge it.
""",
            tools=(
                list(self.rdf_agent.tools) if representation == "RDF"
                else list(self.lpg_agent.tools)
            ),
            output_type=AgentOutputSchema(CritiqueOutput, strict_json_schema=False),
            model=model,
        )

    # ----- Stage 1: Independent generation -----

    async def generate_independently(
        self,
        question: str,
        *,
        experiment_name: str = "debate",
        opik_trace_id: str = "",
    ) -> tuple[RDFAgentOutput, LPGAgentOutput]:
        """Run both agents independently on the same question.

        Both agents query their respective graphs and produce answers
        without seeing each other's output.
        """
        rdf_output, _ = await self._run_agent_with_trace(
            self.rdf_agent,
            question,
            stage_name="run_rdf_agent",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
        )
        lpg_output, _ = await self._run_agent_with_trace(
            self.lpg_agent,
            question,
            stage_name="run_lpg_agent",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
        )
        return rdf_output, lpg_output

    # ----- Stage 2: Compare outputs -----

    def compare_outputs(
        self,
        rdf_output: RDFAgentOutput,
        lpg_output: LPGAgentOutput,
    ) -> ComparisonResult:
        """Analyze agreement/disagreement between RDF and LPG outputs.

        This is a deterministic comparison — no LLM call.
        Checks confidence levels, evidence overlap, and null/ambiguous status.
        """
        confidence_gap = abs(rdf_output.confidence - lpg_output.confidence)

        rdf_null = (
            rdf_output.confidence < CONFIDENCE_THRESHOLD
            or not rdf_output.claims
            or rdf_output.answer_draft.strip() == ""
        )
        lpg_null = (
            lpg_output.confidence < CONFIDENCE_THRESHOLD
            or not lpg_output.claims
            or lpg_output.answer_draft.strip() == ""
        )

        rdf_evidence = _evidence_key_set(rdf_output.evidence_items)
        lpg_evidence = _evidence_key_set(lpg_output.evidence_items)
        if rdf_evidence and lpg_evidence:
            shared_evidence = rdf_evidence & lpg_evidence
            all_evidence = rdf_evidence | lpg_evidence
            evidence_overlap = len(shared_evidence) / len(all_evidence) if all_evidence else 0.0
        else:
            evidence_overlap = 0.0

        # Answers agree if: both have claims, overlap is high, confidence gap is small
        answers_agree = (
            not rdf_null
            and not lpg_null
            and evidence_overlap > 0.4
            and confidence_gap < AGREEMENT_CONFIDENCE_GAP
        )

        rdf_support = rdf_output.retrieval_support_level or _infer_support_level(rdf_output)
        lpg_support = lpg_output.retrieval_support_level or _infer_support_level(lpg_output)

        support_aware_router = self.retrieval_variant == "search_a_v5"

        # Decide recommended strategy
        if support_aware_router and rdf_support == "direct_support" and lpg_support == "no_support":
            strategy = "self_reflection"
            rationale = "RDF has direct support while LPG has no support, so the system prefers a focused self-reflection path before broader escalation."
        elif support_aware_router and lpg_support == "direct_support" and rdf_support == "no_support":
            strategy = "self_reflection"
            rationale = "LPG has direct support while RDF has no support, so the system prefers a focused self-reflection path before broader escalation."
        elif answers_agree:
            strategy = "direct_synthesis"
            rationale = "Both representations produced non-null answers with sufficient evidence overlap and a small confidence gap."
        elif rdf_null != lpg_null:
            # One side is weak — self-reflection might help
            strategy = "self_reflection"
            rationale = "One representation is effectively null or ambiguous while the other has usable evidence, so self-reflection is cheaper than full debate."
        elif not answers_agree and not rdf_null and not lpg_null:
            # Both have answers but they disagree
            strategy = "debate"
            rationale = "Both representations produced usable but conflicting answers, so cross-representation debate may expose complementary strengths."
        else:
            # Both null or ambiguous
            strategy = "judge"
            rationale = "Both representations are weak or ambiguous, so the system must escalate to a guarded synthesis/judge step."

        if rdf_output.retrieval_support_level == "direct_support" and lpg_output.retrieval_support_level != "direct_support":
            rationale += " RDF has direct support while LPG does not."
        elif lpg_output.retrieval_support_level == "direct_support" and rdf_output.retrieval_support_level != "direct_support":
            rationale += " LPG has direct support while RDF does not."

        return ComparisonResult(
            answers_agree=answers_agree,
            confidence_gap=round(confidence_gap, 4),
            evidence_overlap_ratio=round(evidence_overlap, 4),
            rdf_null_or_ambiguous=rdf_null,
            lpg_null_or_ambiguous=lpg_null,
            strategy_rationale=rationale,
            decision_thresholds={
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "agreement_confidence_gap": AGREEMENT_CONFIDENCE_GAP,
                "agreement_evidence_overlap_min": 0.4,
            },
            recommended_strategy=strategy,
        )

    # ----- Stage 3: Decide strategy (returns the comparison's recommendation) -----

    def decide_strategy(self, comparison: ComparisonResult) -> str:
        """Return the resolution strategy. Currently defers to compare_outputs' recommendation.

        Separated as a distinct stage for tracing and potential override logic.
        """
        return comparison.recommended_strategy

    # ----- Stage 4: Execute strategy -----

    async def execute_direct_synthesis(
        self,
        question: str,
        rdf_output: RDFAgentOutput,
        lpg_output: LPGAgentOutput,
        *,
        experiment_name: str = "debate",
        opik_trace_id: str = "",
    ) -> SynthesisOutput:
        """Both agents agree — synthesize directly."""
        synthesis, _ = await self._run_synthesis(
            question,
            rdf_output,
            lpg_output,
            mode="direct_synthesis",
            stage_name="run_synthesis_agent",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
        )
        return synthesis

    async def execute_self_reflection(
        self,
        question: str,
        rdf_output: RDFAgentOutput,
        lpg_output: LPGAgentOutput,
        comparison: ComparisonResult,
        *,
        experiment_name: str = "debate",
        opik_trace_id: str = "",
    ) -> tuple[SynthesisOutput, Optional[CritiqueOutput], Optional[CritiqueOutput]]:
        """One agent is weak — let it reconsider with the other's evidence."""
        rdf_critique = None
        lpg_critique = None

        if comparison.rdf_null_or_ambiguous:
            # RDF was weak — show it the LPG evidence to reconsider
            rdf_critique, _ = await self._run_critique(
                self._rdf_critique_agent,
                question,
                other_output=lpg_output,
                stage_name="run_rdf_critique",
                experiment_name=experiment_name,
                opik_trace_id=opik_trace_id,
                strategy="self_reflection",
            )
        if comparison.lpg_null_or_ambiguous:
            # LPG was weak — show it the RDF evidence
            lpg_critique, _ = await self._run_critique(
                self._lpg_critique_agent,
                question,
                other_output=rdf_output,
                stage_name="run_lpg_critique",
                experiment_name=experiment_name,
                opik_trace_id=opik_trace_id,
                strategy="self_reflection",
            )

        synthesis, _ = await self._run_synthesis(
            question,
            rdf_output,
            lpg_output,
            mode="self_reflection",
            rdf_critique=rdf_critique,
            lpg_critique=lpg_critique,
            stage_name="run_synthesis_agent",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
        )
        return synthesis, rdf_critique, lpg_critique

    async def execute_debate(
        self,
        question: str,
        rdf_output: RDFAgentOutput,
        lpg_output: LPGAgentOutput,
        *,
        experiment_name: str = "debate",
        opik_trace_id: str = "",
    ) -> tuple[SynthesisOutput, CritiqueOutput, CritiqueOutput]:
        """Agents disagree — each critiques the other's claims."""
        rdf_critique, _ = await self._run_critique(
            self._rdf_critique_agent,
            question,
            other_output=lpg_output,
            stage_name="run_rdf_critique",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
            strategy="debate",
        )
        lpg_critique, _ = await self._run_critique(
            self._lpg_critique_agent,
            question,
            other_output=rdf_output,
            stage_name="run_lpg_critique",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
            strategy="debate",
        )

        synthesis, _ = await self._run_synthesis(
            question,
            rdf_output,
            lpg_output,
            mode="debate",
            rdf_critique=rdf_critique,
            lpg_critique=lpg_critique,
            stage_name="run_synthesis_agent",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
        )
        return synthesis, rdf_critique, lpg_critique

    async def execute_judge(
        self,
        question: str,
        rdf_output: RDFAgentOutput,
        lpg_output: LPGAgentOutput,
        *,
        experiment_name: str = "debate",
        opik_trace_id: str = "",
    ) -> tuple[SynthesisOutput, CritiqueOutput, CritiqueOutput]:
        """Persistent disagreement or both weak — critique then judge."""
        rdf_critique, _ = await self._run_critique(
            self._rdf_critique_agent,
            question,
            other_output=lpg_output,
            stage_name="run_rdf_critique",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
            strategy="judge",
        )
        lpg_critique, _ = await self._run_critique(
            self._lpg_critique_agent,
            question,
            other_output=rdf_output,
            stage_name="run_lpg_critique",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
            strategy="judge",
        )

        synthesis, _ = await self._run_synthesis(
            question,
            rdf_output,
            lpg_output,
            mode="judge",
            rdf_critique=rdf_critique,
            lpg_critique=lpg_critique,
            stage_name="run_synthesis_agent",
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
        )
        return synthesis, rdf_critique, lpg_critique

    # ----- Stage 5: Full pipeline orchestrator -----

    async def run_async(
        self,
        question: str,
        *,
        experiment_name: str = "debate",
    ) -> DebateResult:
        """Run the full debate pipeline for one question.

        Pipeline:
          1. generate_independently  → RDFAgentOutput + LPGAgentOutput
          2. compare_outputs         → ComparisonResult
          3. decide_strategy         → str
          4. execute_{strategy}      → SynthesisOutput + optional CritiqueOutputs
          5. assemble DebateResult
        """
        trace_id = ""
        trace_backend = ""
        graph_definition = self._build_agent_graph_definition()
        database_bindings = get_database_bindings()
        question_type = _classify_question_type(question)
        anchor_terms = _extract_anchor_terms(question)
        retrieval_plan = _search_a_plan(question_type)
        rdf_schema_card = get_rdf_schema_card()
        lpg_schema_card = get_lpg_schema_card()
        rdf_prompt = _build_agent_prompt(
            question=question,
            question_type=question_type,
            anchor_terms=anchor_terms,
            retrieval_plan=retrieval_plan,
            schema_card=rdf_schema_card,
            representation="rdf",
        )
        lpg_prompt = _build_agent_prompt(
            question=question,
            question_type=question_type,
            anchor_terms=anchor_terms,
            retrieval_plan=retrieval_plan,
            schema_card=lpg_schema_card,
            representation="lpg",
        )
        openai_traces: dict[str, AgentTraceRecord] = {}
        current_stage = "begin_trace"
        try:
            if self.tracer:
                trace_id = self.tracer.begin_trace(
                    name=_debate_trace_name(experiment_name=experiment_name, question=question),
                    metadata={
                        "experiment_name": experiment_name,
                        "pipeline_family": "debate",
                        "run_kind": _debate_run_kind(experiment_name),
                        "retrieval_scope_mode": "example_scoped",
                        "reference_bound": True,
                        "model": self.model,
                        "question_length": len(question),
                        "question_preview": _truncate_text(question, limit=120),
                        "question_type": question_type,
                        "anchor_terms": anchor_terms[:5],
                        "retrieval_plan": retrieval_plan,
                        "retrieval_variant": self.retrieval_variant,
                        "source_experiment": get_active_scope().get("experiment_name", ""),
                        "scoped_example_id": get_active_scope().get("example_id", ""),
                        "_opik_graph_definition": graph_definition,
                        "trace_family": "debate_dual_tracing",
                        "trace_display_name": _debate_trace_name(experiment_name=experiment_name, question=question),
                        "rdf_database": database_bindings["rdf"],
                        "lpg_database": database_bindings["lpg"],
                    },
                )
                trace_backend = getattr(self.tracer, "backend", "")

            # Stage 1: Independent generation
            current_stage = "run_rdf_agent"
            if self.tracer:
                with self.tracer.span(
                    "run_rdf_agent",
                    inputs={
                        "question_preview": _truncate_text(question),
                        "question_length": len(question),
                        "question_type": question_type,
                        "anchor_terms": anchor_terms[:5],
                        "retrieval_plan": retrieval_plan,
                        "retrieval_variant": self.retrieval_variant,
                        "retrieval_scope_mode": "example_scoped",
                        "reference_bound": True,
                    },
                    metadata={
                        "representation": "rdf",
                        "target_database": database_bindings["rdf"],
                        "retrieval_variant": self.retrieval_variant,
                        "retrieval_scope_mode": "example_scoped",
                        "reference_bound": True,
                    },
                ) as s:
                    rdf_output, rdf_trace = await self._run_agent_with_trace(
                        self.rdf_agent,
                        rdf_prompt,
                        stage_name="run_rdf_agent",
                        experiment_name=experiment_name,
                        opik_trace_id=trace_id,
                        target_database=database_bindings["rdf"],
                    )
                    rdf_output = _backfill_retrieval_fields(
                        rdf_output,
                        question_type=question_type,
                        anchor_terms=anchor_terms,
                        retrieval_plan=retrieval_plan,
                    )
                    openai_traces["run_rdf_agent"] = rdf_trace
                    s["outputs"] = {
                        "agent_name": self.rdf_agent.name,
                        "rdf_claims": len(rdf_output.claims),
                        "rdf_evidence_items": len(rdf_output.evidence_items),
                        "rdf_tool_trace_count": len(rdf_output.tool_trace_digest),
                        "question_type": rdf_output.question_type or question_type,
                        "retrieval_support_level": rdf_output.retrieval_support_level,
                        "retrieval_variant": self.retrieval_variant,
                        "rdf_confidence": rdf_output.confidence,
                        "target_database": database_bindings["rdf"],
                        "openai_trace_id": rdf_trace.openai_trace_id,
                        "openai_response_id": rdf_trace.openai_response_id,
                        "answer_preview": _truncate_text(rdf_output.answer_draft),
                        "tool_trace_digest": _tool_digest_summary(rdf_output.tool_trace_digest),
                    }
                current_stage = "run_lpg_agent"
                with self.tracer.span(
                    "run_lpg_agent",
                    inputs={
                        "question_preview": _truncate_text(question),
                        "question_length": len(question),
                        "question_type": question_type,
                        "anchor_terms": anchor_terms[:5],
                        "retrieval_plan": retrieval_plan,
                        "retrieval_variant": self.retrieval_variant,
                        "retrieval_scope_mode": "example_scoped",
                        "reference_bound": True,
                    },
                    metadata={
                        "representation": "lpg",
                        "target_database": database_bindings["lpg"],
                        "retrieval_variant": self.retrieval_variant,
                        "retrieval_scope_mode": "example_scoped",
                        "reference_bound": True,
                    },
                ) as s:
                    lpg_output, lpg_trace = await self._run_agent_with_trace(
                        self.lpg_agent,
                        lpg_prompt,
                        stage_name="run_lpg_agent",
                        experiment_name=experiment_name,
                        opik_trace_id=trace_id,
                        target_database=database_bindings["lpg"],
                    )
                    lpg_output = _backfill_retrieval_fields(
                        lpg_output,
                        question_type=question_type,
                        anchor_terms=anchor_terms,
                        retrieval_plan=retrieval_plan,
                    )
                    openai_traces["run_lpg_agent"] = lpg_trace
                    s["outputs"] = {
                        "agent_name": self.lpg_agent.name,
                        "lpg_claims": len(lpg_output.claims),
                        "lpg_evidence_items": len(lpg_output.evidence_items),
                        "lpg_tool_trace_count": len(lpg_output.tool_trace_digest),
                        "question_type": lpg_output.question_type or question_type,
                        "retrieval_support_level": lpg_output.retrieval_support_level,
                        "retrieval_variant": self.retrieval_variant,
                        "lpg_confidence": lpg_output.confidence,
                        "target_database": database_bindings["lpg"],
                        "openai_trace_id": lpg_trace.openai_trace_id,
                        "openai_response_id": lpg_trace.openai_response_id,
                        "answer_preview": _truncate_text(lpg_output.answer_draft),
                        "tool_trace_digest": _tool_digest_summary(lpg_output.tool_trace_digest),
                    }
            else:
                rdf_output, rdf_trace = await self._run_agent_with_trace(
                    self.rdf_agent,
                    rdf_prompt,
                    stage_name="run_rdf_agent",
                    experiment_name=experiment_name,
                    opik_trace_id=trace_id,
                    target_database=database_bindings["rdf"],
                )
                rdf_output = _backfill_retrieval_fields(
                    rdf_output,
                    question_type=question_type,
                    anchor_terms=anchor_terms,
                    retrieval_plan=retrieval_plan,
                )
                current_stage = "run_lpg_agent"
                lpg_output, lpg_trace = await self._run_agent_with_trace(
                    self.lpg_agent,
                    lpg_prompt,
                    stage_name="run_lpg_agent",
                    experiment_name=experiment_name,
                    opik_trace_id=trace_id,
                    target_database=database_bindings["lpg"],
                )
                lpg_output = _backfill_retrieval_fields(
                    lpg_output,
                    question_type=question_type,
                    anchor_terms=anchor_terms,
                    retrieval_plan=retrieval_plan,
                )
                openai_traces["run_rdf_agent"] = rdf_trace
                openai_traces["run_lpg_agent"] = lpg_trace

            # Stage 2: Compare
            current_stage = "compare_outputs"
            if self.tracer:
                with self.tracer.span("compare_outputs", inputs={
                    "rdf_confidence": rdf_output.confidence,
                    "lpg_confidence": lpg_output.confidence,
                }) as s:
                    comparison = self.compare_outputs(rdf_output, lpg_output)
                    s["outputs"] = comparison.model_dump()
            else:
                comparison = self.compare_outputs(rdf_output, lpg_output)

            # Stage 3: Decide strategy
            current_stage = "decide_strategy"
            if self.tracer:
                with self.tracer.span(
                    "decide_strategy",
                    inputs={
                        "answers_agree": comparison.answers_agree,
                        "confidence_gap": comparison.confidence_gap,
                        "evidence_overlap_ratio": comparison.evidence_overlap_ratio,
                        "rdf_null_or_ambiguous": comparison.rdf_null_or_ambiguous,
                        "lpg_null_or_ambiguous": comparison.lpg_null_or_ambiguous,
                    },
                ) as s:
                    strategy = self.decide_strategy(comparison)
                    s["outputs"] = {
                        "strategy": strategy,
                        "strategy_rationale": comparison.strategy_rationale,
                        "decision_thresholds": comparison.decision_thresholds,
                    }
            else:
                strategy = self.decide_strategy(comparison)

            # Stage 4: Execute strategy
            rdf_critique: Optional[CritiqueOutput] = None
            lpg_critique: Optional[CritiqueOutput] = None

            if self.tracer:
                if strategy == "self_reflection":
                    if comparison.rdf_null_or_ambiguous:
                        current_stage = "run_rdf_critique"
                        with self.tracer.span(
                            "run_rdf_critique",
                            inputs={"question_preview": _truncate_text(question), "other_representation": "lpg"},
                            metadata={"strategy": strategy, "representation": "rdf", "target_database": database_bindings["rdf"]},
                        ) as s:
                            rdf_critique, critique_trace = await self._run_critique(
                                self._rdf_critique_agent,
                                question,
                                other_output=lpg_output,
                                stage_name="run_rdf_critique",
                                experiment_name=experiment_name,
                                opik_trace_id=trace_id,
                                strategy=strategy,
                                target_database=database_bindings["rdf"],
                            )
                            openai_traces["run_rdf_critique"] = critique_trace
                            s["outputs"] = {
                                "openai_trace_id": critique_trace.openai_trace_id,
                                "openai_response_id": critique_trace.openai_response_id,
                                "revised_confidence": rdf_critique.revised_confidence,
                                "revised_answer_preview": _truncate_text(rdf_critique.revised_answer),
                            }
                    if comparison.lpg_null_or_ambiguous:
                        current_stage = "run_lpg_critique"
                        with self.tracer.span(
                                "run_lpg_critique",
                                inputs={"question_preview": _truncate_text(question), "other_representation": "rdf"},
                                metadata={"strategy": strategy, "representation": "lpg", "target_database": database_bindings["lpg"]},
                            ) as s:
                            lpg_critique, critique_trace = await self._run_critique(
                                self._lpg_critique_agent,
                                question,
                                other_output=rdf_output,
                                stage_name="run_lpg_critique",
                                experiment_name=experiment_name,
                                opik_trace_id=trace_id,
                                strategy=strategy,
                                target_database=database_bindings["lpg"],
                            )
                            openai_traces["run_lpg_critique"] = critique_trace
                            s["outputs"] = {
                                "openai_trace_id": critique_trace.openai_trace_id,
                                "openai_response_id": critique_trace.openai_response_id,
                                "revised_confidence": lpg_critique.revised_confidence,
                                "revised_answer_preview": _truncate_text(lpg_critique.revised_answer),
                            }
                elif strategy in {"debate", "judge"}:
                    current_stage = "run_rdf_critique"
                    with self.tracer.span(
                        "run_rdf_critique",
                        inputs={"question_preview": _truncate_text(question), "other_representation": "lpg"},
                        metadata={"strategy": strategy, "representation": "rdf", "target_database": database_bindings["rdf"]},
                    ) as s:
                        rdf_critique, critique_trace = await self._run_critique(
                            self._rdf_critique_agent,
                            question,
                            other_output=lpg_output,
                            stage_name="run_rdf_critique",
                            experiment_name=experiment_name,
                            opik_trace_id=trace_id,
                            strategy=strategy,
                            target_database=database_bindings["rdf"],
                        )
                        openai_traces["run_rdf_critique"] = critique_trace
                        s["outputs"] = {
                            "openai_trace_id": critique_trace.openai_trace_id,
                            "openai_response_id": critique_trace.openai_response_id,
                            "revised_confidence": rdf_critique.revised_confidence,
                            "revised_answer_preview": _truncate_text(rdf_critique.revised_answer),
                        }
                    current_stage = "run_lpg_critique"
                    with self.tracer.span(
                        "run_lpg_critique",
                        inputs={"question_preview": _truncate_text(question), "other_representation": "rdf"},
                        metadata={"strategy": strategy, "representation": "lpg", "target_database": database_bindings["lpg"]},
                    ) as s:
                        lpg_critique, critique_trace = await self._run_critique(
                            self._lpg_critique_agent,
                            question,
                            other_output=rdf_output,
                            stage_name="run_lpg_critique",
                            experiment_name=experiment_name,
                            opik_trace_id=trace_id,
                            strategy=strategy,
                            target_database=database_bindings["lpg"],
                        )
                        openai_traces["run_lpg_critique"] = critique_trace
                        s["outputs"] = {
                            "openai_trace_id": critique_trace.openai_trace_id,
                            "openai_response_id": critique_trace.openai_response_id,
                            "revised_confidence": lpg_critique.revised_confidence,
                            "revised_answer_preview": _truncate_text(lpg_critique.revised_answer),
                        }

                current_stage = "run_synthesis_agent"
                with self.tracer.span(
                    "run_synthesis_agent",
                    inputs={
                        "strategy": strategy,
                        "rdf_claims": len(rdf_output.claims),
                        "lpg_claims": len(lpg_output.claims),
                        "has_rdf_critique": rdf_critique is not None,
                        "has_lpg_critique": lpg_critique is not None,
                    },
                    metadata={"strategy": strategy},
                ) as s:
                    synthesis, synthesis_trace = await self._run_synthesis(
                        question,
                        rdf_output,
                        lpg_output,
                        mode=strategy,
                        rdf_critique=rdf_critique,
                        lpg_critique=lpg_critique,
                        stage_name="run_synthesis_agent",
                        experiment_name=experiment_name,
                        opik_trace_id=trace_id,
                        target_database="hybrid",
                    )
                    openai_traces["run_synthesis_agent"] = synthesis_trace
                    s["outputs"] = {
                        "openai_trace_id": synthesis_trace.openai_trace_id,
                        "openai_response_id": synthesis_trace.openai_response_id,
                        "resolution_mode": synthesis.resolution_mode,
                        "final_confidence": synthesis.final_confidence,
                        "selected_representation": synthesis.selected_supporting_representation,
                        "representation_condition": "representation_preserving_joint",
                        "final_answer_preview": _truncate_text(synthesis.final_answer),
                    }
            else:
                if strategy == "self_reflection":
                    if comparison.rdf_null_or_ambiguous:
                        current_stage = "run_rdf_critique"
                        rdf_critique, critique_trace = await self._run_critique(
                            self._rdf_critique_agent,
                            question,
                            other_output=lpg_output,
                            stage_name="run_rdf_critique",
                            experiment_name=experiment_name,
                            opik_trace_id=trace_id,
                            strategy=strategy,
                            target_database=database_bindings["rdf"],
                        )
                        openai_traces["run_rdf_critique"] = critique_trace
                    if comparison.lpg_null_or_ambiguous:
                        current_stage = "run_lpg_critique"
                        lpg_critique, critique_trace = await self._run_critique(
                            self._lpg_critique_agent,
                            question,
                            other_output=rdf_output,
                            stage_name="run_lpg_critique",
                            experiment_name=experiment_name,
                            opik_trace_id=trace_id,
                            strategy=strategy,
                            target_database=database_bindings["lpg"],
                        )
                        openai_traces["run_lpg_critique"] = critique_trace
                elif strategy in {"debate", "judge"}:
                    current_stage = "run_rdf_critique"
                    rdf_critique, critique_trace = await self._run_critique(
                        self._rdf_critique_agent,
                        question,
                        other_output=lpg_output,
                        stage_name="run_rdf_critique",
                        experiment_name=experiment_name,
                        opik_trace_id=trace_id,
                        strategy=strategy,
                        target_database=database_bindings["rdf"],
                    )
                    openai_traces["run_rdf_critique"] = critique_trace
                    current_stage = "run_lpg_critique"
                    lpg_critique, critique_trace = await self._run_critique(
                        self._lpg_critique_agent,
                        question,
                        other_output=rdf_output,
                        stage_name="run_lpg_critique",
                        experiment_name=experiment_name,
                        opik_trace_id=trace_id,
                        strategy=strategy,
                        target_database=database_bindings["lpg"],
                    )
                    openai_traces["run_lpg_critique"] = critique_trace

            current_stage = "run_synthesis_agent"
            synthesis, synthesis_trace = await self._run_synthesis(
                question,
                rdf_output,
                lpg_output,
                mode=strategy,
                rdf_critique=rdf_critique,
                lpg_critique=lpg_critique,
                stage_name="run_synthesis_agent",
                experiment_name=experiment_name,
                opik_trace_id=trace_id,
                target_database="hybrid",
            )
            openai_traces["run_synthesis_agent"] = synthesis_trace
        except Exception as exc:
            raise DebateExecutionError(
                stage_name=current_stage,
                original_exception=exc,
                experiment_name=experiment_name,
                target_databases=database_bindings,
                active_scope=get_active_scope(),
            ) from exc

        # Build handoff log for tracing
        handoff_log = HandoffLog(
            initial_answer_agreement=comparison.answers_agree,
            initial_confidence_gap=comparison.confidence_gap,
            initial_evidence_overlap=comparison.evidence_overlap_ratio,
            rdf_null_or_ambiguous=comparison.rdf_null_or_ambiguous,
            lpg_null_or_ambiguous=comparison.lpg_null_or_ambiguous,
            self_reflection_trigger=(strategy == "self_reflection"),
            debate_trigger=(strategy == "debate"),
            judge_trigger=(strategy == "judge"),
            strategy_rationale=comparison.strategy_rationale,
            decision_thresholds=comparison.decision_thresholds,
            final_resolution_mode=strategy,
        )

        trace_linkage = DebateTraceLinkage(
            opik_trace_id=trace_id,
            opik_backend=trace_backend,
            opik_graph_definition=graph_definition,
            openai_traces=openai_traces,
        )

        if self.tracer:
            self.tracer.log_score("final_confidence", synthesis.final_confidence)
            self.tracer.log_score("initial_agreement", 1.0 if comparison.answers_agree else 0.0)
            self.tracer.end_trace(
                output={
                    "strategy": strategy,
                    "strategy_rationale": comparison.strategy_rationale,
                    "decision_thresholds": comparison.decision_thresholds,
                    "final_confidence": synthesis.final_confidence,
                    "selected_supporting_representation": synthesis.selected_supporting_representation,
                    "representation_condition": "representation_preserving_joint",
                    "openai_trace_stages": sorted(openai_traces.keys()),
                },
                metadata={
                    "opik_backend": trace_backend,
                    "openai_trace_ids": {
                        stage: record.openai_trace_id
                        for stage, record in openai_traces.items()
                    },
                    "target_databases": database_bindings,
                    "openai_response_ids": {
                        stage: record.openai_response_id
                        for stage, record in openai_traces.items()
                    },
                    "final_resolution_mode": strategy,
                },
            )

        return DebateResult(
            question=question,
            representation_condition="representation_preserving_joint",
            rdf_output=rdf_output,
            lpg_output=lpg_output,
            comparison=comparison,
            handoff_log=handoff_log,
            synthesis=synthesis,
            rdf_critique=rdf_critique,
            lpg_critique=lpg_critique,
            trace_id=trace_id,
            trace_linkage=trace_linkage,
        )

    def run(self, question: str, **kwargs: Any) -> DebateResult:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(question, **kwargs))

    # ----- Internal helpers -----

    async def _run_critique(
        self,
        critique_agent: Agent,
        question: str,
        other_output: RDFAgentOutput | LPGAgentOutput,
        *,
        stage_name: str,
        experiment_name: str,
        opik_trace_id: str,
        strategy: str,
        target_database: str,
    ) -> tuple[CritiqueOutput, AgentTraceRecord]:
        """Run a critique agent on the other side's output."""
        critique_prompt = (
            f"Original question: {question}\n\n"
            f"The other agent ({other_output.representation.upper()}) produced:\n"
            f"Answer: {other_output.answer_draft}\n\n"
            f"Claims:\n"
        )
        for claim in other_output.claims:
            critique_prompt += f"- [{claim.claim_id}] {claim.text} (confidence={claim.confidence}, status={claim.status})\n"

        critique_prompt += f"\nEvidence:\n"
        for ev in other_output.evidence_items:
            critique_prompt += f"- [{ev.evidence_id}] {ev.content} (source: {ev.source_ref})\n"
        if getattr(other_output, "tool_trace_digest", None):
            critique_prompt += "\nTool trace digest:\n"
            for item in other_output.tool_trace_digest[:5]:
                critique_prompt += (
                    f"- tool={item.tool_name} | intent={item.query_intent} | "
                    f"found={item.results_summary} | analysis={item.analysis}\n"
                )

        critique_prompt += (
            f"\nMissing evidence noted by the other agent: {', '.join(other_output.missing_evidence) or 'none'}\n\n"
            f"Critique each claim using your own graph database. "
            f"Then produce a revised answer that accounts for valid points."
        )

        return await self._run_agent_with_trace(
            critique_agent,
            critique_prompt,
            stage_name=stage_name,
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
            strategy=strategy,
            other_representation=other_output.representation,
            target_database=target_database,
        )

    async def _run_synthesis(
        self,
        question: str,
        rdf_output: RDFAgentOutput,
        lpg_output: LPGAgentOutput,
        *,
        mode: str,
        rdf_critique: Optional[CritiqueOutput] = None,
        lpg_critique: Optional[CritiqueOutput] = None,
        stage_name: str,
        experiment_name: str,
        opik_trace_id: str,
        target_database: str,
    ) -> tuple[SynthesisOutput, AgentTraceRecord]:
        """Run the synthesis agent to produce the final answer."""
        synthesis_prompt = f"Question: {question}\n\n"
        synthesis_prompt += "Representation condition: representation-preserving joint view.\n"
        synthesis_prompt += "Do not merge RDF and LPG into one monolithic graph in your reasoning; keep provenance attributable.\n\n"

        # RDF agent section
        synthesis_prompt += "=== RDF Agent Output ===\n"
        synthesis_prompt += f"Answer: {rdf_output.answer_draft}\n"
        synthesis_prompt += f"Confidence: {rdf_output.confidence}\n"
        synthesis_prompt += f"Claims ({len(rdf_output.claims)}):\n"
        for c in rdf_output.claims:
            synthesis_prompt += f"  [{c.claim_id}] {c.text} (conf={c.confidence}, status={c.status})\n"
        if rdf_output.uncertainty_notes:
            synthesis_prompt += f"Uncertainties: {', '.join(rdf_output.uncertainty_notes)}\n"
        if rdf_output.missing_evidence:
            synthesis_prompt += f"Missing evidence: {', '.join(rdf_output.missing_evidence)}\n"
        if rdf_output.evidence_items:
            synthesis_prompt += "Evidence:\n"
            for ev in rdf_output.evidence_items:
                synthesis_prompt += (
                    f"  [{ev.evidence_id}] {ev.content} "
                    f"(source={ev.source_ref}, type={ev.source_type})\n"
                )
        if rdf_output.tool_trace_digest:
            synthesis_prompt += "Tool trace digest:\n"
            for item in rdf_output.tool_trace_digest[:5]:
                synthesis_prompt += (
                    f"  tool={item.tool_name} | intent={item.query_intent} | "
                    f"found={item.results_summary} | analysis={item.analysis}\n"
                )

        # LPG agent section
        synthesis_prompt += "\n=== LPG Agent Output ===\n"
        synthesis_prompt += f"Answer: {lpg_output.answer_draft}\n"
        synthesis_prompt += f"Confidence: {lpg_output.confidence}\n"
        synthesis_prompt += f"Claims ({len(lpg_output.claims)}):\n"
        for c in lpg_output.claims:
            synthesis_prompt += f"  [{c.claim_id}] {c.text} (conf={c.confidence}, status={c.status})\n"
        if lpg_output.uncertainty_notes:
            synthesis_prompt += f"Uncertainties: {', '.join(lpg_output.uncertainty_notes)}\n"
        if lpg_output.missing_evidence:
            synthesis_prompt += f"Missing evidence: {', '.join(lpg_output.missing_evidence)}\n"
        if lpg_output.evidence_items:
            synthesis_prompt += "Evidence:\n"
            for ev in lpg_output.evidence_items:
                synthesis_prompt += (
                    f"  [{ev.evidence_id}] {ev.content} "
                    f"(source={ev.source_ref}, type={ev.source_type})\n"
                )
        if lpg_output.tool_trace_digest:
            synthesis_prompt += "Tool trace digest:\n"
            for item in lpg_output.tool_trace_digest[:5]:
                synthesis_prompt += (
                    f"  tool={item.tool_name} | intent={item.query_intent} | "
                    f"found={item.results_summary} | analysis={item.analysis}\n"
                )

        # Critique sections if available
        if rdf_critique:
            synthesis_prompt += "\n=== RDF Agent Critique of LPG ===\n"
            synthesis_prompt += f"Revised answer: {rdf_critique.revised_answer}\n"
            synthesis_prompt += f"Revised confidence: {rdf_critique.revised_confidence}\n"
            synthesis_prompt += f"Conceded: {', '.join(rdf_critique.conceded_points) or 'none'}\n"
            synthesis_prompt += f"Maintained: {', '.join(rdf_critique.maintained_points) or 'none'}\n"

        if lpg_critique:
            synthesis_prompt += "\n=== LPG Agent Critique of RDF ===\n"
            synthesis_prompt += f"Revised answer: {lpg_critique.revised_answer}\n"
            synthesis_prompt += f"Revised confidence: {lpg_critique.revised_confidence}\n"
            synthesis_prompt += f"Conceded: {', '.join(lpg_critique.conceded_points) or 'none'}\n"
            synthesis_prompt += f"Maintained: {', '.join(lpg_critique.maintained_points) or 'none'}\n"

        synthesis_prompt += f"\nResolution mode: {mode}\n"
        synthesis_prompt += "Evaluate ontology consistency and evidence richness separately, then synthesize the final answer with explicit attribution."

        return await self._run_agent_with_trace(
            self.synthesis_agent,
            synthesis_prompt,
            stage_name=stage_name,
            experiment_name=experiment_name,
            opik_trace_id=opik_trace_id,
            strategy=mode,
            target_database=target_database,
        )
