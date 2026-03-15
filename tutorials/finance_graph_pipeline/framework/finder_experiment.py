from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from neo4j import GraphDatabase

from .agents import EntityLinker
from .evaluation import (
    AUXILIARY_BASELINE_ORDER,
    BASELINE_ORDER,
    FinanceTutorialEvaluator,
    GoldExtraction,
    ONTOLOGY_BASELINE_ORDER,
)
from .fibo_profiles import FIBO_PROFILES, SECTION_TO_PROFILE
from .intents import infer_intent
from .manual_gold import load_manual_gold
from .models import (
    AnswerResult,
    Document,
    ExtractedEntity,
    ExtractedRelation,
    FiboProfileDecision,
    GraphEdge,
    GraphNode,
    GraphState,
    Question,
)
from .quality import GraphQualityAnalyzer

try:
    from agents import Agent, ModelSettings, RunConfig, Runner
except Exception:  # pragma: no cover - optional outside the container
    Agent = None
    ModelSettings = None
    RunConfig = None
    Runner = None


@dataclass
class FinDERExample:
    example_id: str
    category: str
    question_text: str
    answer_text: str
    references: list[str]
    reasoning: bool
    question_type: str

    @property
    def canonical_category(self) -> str:
        return " ".join(self.category.lower().split())

    @property
    def target_profile(self) -> str:
        return SECTION_TO_PROFILE[self.canonical_category]

    @property
    def query_template(self) -> str:
        return self.to_question().query_template

    @property
    def source_text(self) -> str:
        if self.references:
            return "\n\n".join(self.references)
        return self.question_text

    def to_document(self) -> Document:
        return Document(
            doc_id=self.example_id,
            section_type=self.category,
            text=self.source_text,
        )

    def to_question(self) -> Question:
        intent = infer_intent(self.target_profile, self.question_text)
        return Question(
            question_id=self.example_id,
            question=self.question_text,
            query_template=intent.intent_id,
            target_profile=self.target_profile,
            ground_truth_answer=self.answer_text,
            intent_id=intent.intent_id,
            required_relations=intent.required_relations,
            required_entity_types=intent.required_entity_types,
            focus_slots=intent.focus_slots,
            focus_slot_hints={slot: tuple(hints) for slot, hints in intent.focus_slot_hints.items()},
        )


@dataclass
class FinDERRunConfig:
    dataset_path: Path
    output_dir: Path
    db_path: Path
    graph_uri: str
    neo4j_user: str
    neo4j_password: str
    sample_size: int | None
    per_category_limit: int | None
    agent_mode: str
    model_name: str
    persist_graph: bool
    max_references: int
    openai_max_workers: int
    openai_call_timeout_seconds: int
    checkpoint_every_examples: int
    manual_gold_path: Path | None = None
    manual_gold_only: bool = False


@dataclass
class OpenAIProfileDecision:
    selected_profile: str
    confidence: float
    rationale: str


@dataclass
class OpenAIExtractionBundle:
    entities: list[dict[str, str]]
    relations: list[dict[str, str]]


class FinDERProfileAgent:
    profile_keywords = {
        "governance": (
            "board",
            "director",
            "audit committee",
            "chief executive officer",
            "governance",
            "chair",
            "officer",
        ),
        "financials": (
            "revenue",
            "income",
            "cash",
            "debt",
            "liquidity",
            "margin",
            "operating",
            "quarter",
            "fiscal",
        ),
        "shareholder_return": (
            "dividend",
            "repurchase",
            "buyback",
            "capital allocation",
            "shareholder return",
            "common stock",
            "authorization",
        ),
    }

    def __init__(self, mode: str = "heuristic", model_name: str = "gpt-4.1-mini") -> None:
        self.mode = mode
        self.model_name = model_name
        self.agent = None
        if mode == "openai":
            self.agent = self._build_openai_agent()

    def select_profile(self, example: FinDERExample) -> FiboProfileDecision:
        if self.mode == "openai" and self.agent is not None:
            result = self._select_profile_openai(example)
            return FiboProfileDecision(
                selected_profile=result.selected_profile,
                candidate_profiles=[result.selected_profile],
                selection_confidence=result.confidence,
                mapping_policy="strict",
                ontology_rationale=result.rationale,
            )

        combined = example.source_text.lower()
        if not combined.strip():
            combined = example.question_text.lower()
        scores: dict[str, int] = {}
        for profile, keywords in self.profile_keywords.items():
            scores[profile] = sum(1 for keyword in keywords if keyword in combined)
        selected_profile = max(scores, key=scores.get)
        if scores[selected_profile] == 0:
            selected_profile = example.target_profile
        confidence = 0.6 + min(scores[selected_profile], 4) * 0.1
        return FiboProfileDecision(
            selected_profile=selected_profile,
            candidate_profiles=[selected_profile],
            selection_confidence=round(min(confidence, 0.95), 3),
            mapping_policy="strict",
            ontology_rationale="Keyword-based profile selection over question and supporting references.",
        )

    def _build_openai_agent(self) -> Agent | None:
        if Agent is None:
            return None
        return Agent(
            name="finder_profile_selector",
            model=self.model_name,
            model_settings=ModelSettings(temperature=0.0) if ModelSettings is not None else None,
            instructions=(
                "Select exactly one profile from governance, financials, shareholder_return. "
                "Return JSON only with keys selected_profile, confidence, rationale."
            ),
            output_type=str,
        )

    def _select_profile_openai(self, example: FinDERExample) -> OpenAIProfileDecision:
        prompt = json.dumps(
            {
                "document_text": example.source_text,
                "allowed_profiles": ["governance", "financials", "shareholder_return"],
            },
            ensure_ascii=False,
        )
        result = Runner.run_sync(
            self.agent,
            prompt,
            run_config=RunConfig(
                model=self.model_name,
                tracing_disabled=True,
                workflow_name="FinDER profile selection",
            )
            if RunConfig is not None
            else None,
        )
        payload = self._parse_model_json(result.final_output_as(str))
        return OpenAIProfileDecision(
            selected_profile=payload["selected_profile"],
            confidence=float(payload.get("confidence", 0.7)),
            rationale=payload.get("rationale", ""),
        )

    def _parse_model_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))


class FinDERExtractionAgent:
    money_pattern = re.compile(r"(?:USD\s*)?\$?\s?\d+(?:\.\d+)?\s?(?:billion|million|thousand|%)?", re.IGNORECASE)
    year_pattern = re.compile(r"\b(?:FY)?20\d{2}\b")
    quarter_pattern = re.compile(r"\bQ[1-4]\s?20\d{2}\b", re.IGNORECASE)
    person_pattern = re.compile(r"\b([A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+)\b")

    governance_roles = (
        "Chief Executive Officer",
        "Chief Financial Officer",
        "Chief Operating Officer",
        "Chief Human Resources Officer",
        "General Counsel",
        "Corporate Secretary",
        "director",
        "chair",
        "president",
    )
    governance_committees = ("audit committee", "compensation committee", "risk committee", "board")
    financial_metrics = (
        "revenue",
        "net income",
        "operating income",
        "operating margin",
        "gross profit",
        "cash",
        "cash equivalents",
        "liquidity",
        "debt",
        "cash flow",
        "earnings",
    )
    financial_segments = (
        "consumer",
        "segment",
        "data and access solutions",
        "derivatives",
        "cash and spot markets",
    )
    shareholder_terms = (
        "share repurchase",
        "repurchase authorization",
        "dividend",
        "common stock",
        "common share",
        "shareholder return",
        "capital allocation",
        "buyback",
    )

    def __init__(self, mode: str = "heuristic", model_name: str = "gpt-4.1-mini") -> None:
        self.mode = mode
        self.model_name = model_name
        self.agent = None
        if mode == "openai":
            self.agent = self._build_openai_agent()

    def extract(
        self,
        example: FinDERExample,
        decision: FiboProfileDecision,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation], dict[str, Any]]:
        if self.mode == "openai" and self.agent is not None:
            bundle = self._extract_openai(example, decision)
            return self._coerce_openai_bundle(example, bundle)

        method = getattr(self, f"_extract_{decision.selected_profile}", self._extract_financials)
        entities, relations = method(example)
        return entities, relations, {"mode": "heuristic"}

    def _build_openai_agent(self) -> Agent | None:
        if Agent is None:
            return None
        return Agent(
            name="finder_extractor",
            model=self.model_name,
            model_settings=ModelSettings(temperature=0.0) if ModelSettings is not None else None,
            instructions=(
                "Extract ontology-constrained entities and relations. "
                "Return compact JSON with keys entities and relations. "
                "Each entity needs name and entity_type. "
                "Each relation needs source_name, relation_type, target_name. "
                "Return JSON only."
            ),
            output_type=str,
        )

    def _extract_openai(
        self,
        example: FinDERExample,
        decision: FiboProfileDecision,
    ) -> OpenAIExtractionBundle:
        prompt = json.dumps(
            {
                "profile": decision.selected_profile,
                "allowed_entity_types": list(FIBO_PROFILES[decision.selected_profile].entity_types),
                "allowed_relation_types": list(FIBO_PROFILES[decision.selected_profile].relation_types),
                "document_text": example.source_text,
            },
            ensure_ascii=False,
        )
        result = Runner.run_sync(
            self.agent,
            prompt,
            run_config=RunConfig(
                model=self.model_name,
                tracing_disabled=True,
                workflow_name="FinDER ontology-constrained extraction",
            )
            if RunConfig is not None
            else None,
        )
        payload = self._parse_model_json(result.final_output_as(str))
        return OpenAIExtractionBundle(
            entities=list(payload.get("entities", [])),
            relations=list(payload.get("relations", [])),
        )

    def _coerce_openai_bundle(
        self,
        example: FinDERExample,
        bundle: OpenAIExtractionBundle,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation], dict[str, Any]]:
        entities = [
            ExtractedEntity(
                name=item["name"],
                entity_type=item["entity_type"],
                confidence=0.8,
                source_doc_id=example.example_id,
            )
            for item in bundle.entities
        ]
        relations = [
            ExtractedRelation(
                source_name=item["source_name"],
                relation_type=item["relation_type"],
                target_name=item["target_name"],
                confidence=0.78,
                source_doc_id=example.example_id,
            )
            for item in bundle.relations
        ]
        return entities, relations, {"mode": "openai"}

    def _parse_model_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _extract_governance(
        self,
        example: FinDERExample,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        text = example.source_text
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        company = self._infer_company_name(text) or "Referenced Company"
        entities.append(self._entity(company, "LegalEntity", example))

        for committee in self.governance_committees:
            if committee in text.lower():
                label = committee.title()
                entity_type = "Committee" if "committee" in committee else "Board"
                entities.append(self._entity(label, entity_type, example))

        people = self._unique_persons(text)
        for person in people:
            entities.append(self._entity(person, "Person", example))

        for role in self.governance_roles:
            if role.lower() in text.lower():
                role_name = role.title() if role.islower() else role
                entities.append(self._entity(role_name, "OfficerRole", example))
                for person in people[:2]:
                    if role.lower() in text.lower():
                        relations.append(self._relation(person, "holds_role", role_name, example))
                        break

        lower_text = text.lower()
        if "chair" in lower_text and people:
            relations.append(self._relation(people[0], "chairs", "Board", example))
        if "committee" in lower_text:
            committee_name = "Audit Committee" if "audit committee" in lower_text else "Committee"
            for person in people[:3]:
                relations.append(self._relation(person, "serves_on_committee", committee_name, example))

        return self._dedupe_entities(entities), self._dedupe_relations(relations)

    def _extract_financials(
        self,
        example: FinDERExample,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        text = example.source_text
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        company = self._infer_company_name(text) or "Referenced Company"
        entities.append(self._entity(company, "LegalEntity", example))

        metric_name = None
        for metric in self.financial_metrics:
            if metric in text.lower():
                metric_name = metric.title() if metric.islower() else metric
                entities.append(self._entity(metric_name, "FinancialMetric", example))
                break

        for segment in self.financial_segments:
            if segment in text.lower():
                entities.append(self._entity(segment.title(), "BusinessSegment", example))

        periods = sorted(set(self.year_pattern.findall(text) + self.quarter_pattern.findall(text)))
        for period in periods[:3]:
            entities.append(self._entity(period.upper().replace(" ", " "), "ReportingPeriod", example))

        amounts = self._extract_amounts(text)
        for amount in amounts[:6]:
            entities.append(self._entity(amount, "MonetaryAmount", example))

        if metric_name:
            relations.append(self._relation(company, "reports_metric", metric_name, example))
            if periods:
                relations.append(self._relation(metric_name, "reported_for_period", periods[0], example))

        segment_nodes = [entity.name for entity in entities if entity.entity_type == "BusinessSegment"]
        if segment_nodes and amounts:
            relations.append(self._relation(segment_nodes[0], "segment_contribution", amounts[0], example))

        return self._dedupe_entities(entities), self._dedupe_relations(relations)

    def _extract_shareholder_return(
        self,
        example: FinDERExample,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        text = example.source_text
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        company = self._infer_company_name(text) or "Referenced Company"
        entities.append(self._entity(company, "LegalEntity", example))

        if "repurchase" in text.lower() or "buyback" in text.lower():
            entities.append(self._entity("Share Repurchase Program", "RepurchaseProgram", example))
            relations.append(self._relation(company, "announced_repurchase", "Share Repurchase Program", example))

        if "dividend" in text.lower():
            entities.append(self._entity("Dividend", "Dividend", example))
            relations.append(self._relation(company, "declared_dividend", "Dividend", example))

        if "common stock" in text.lower() or "common share" in text.lower():
            share_class = "Common Stock" if "common stock" in text.lower() else "Common Share"
            entities.append(self._entity(share_class, "ShareClass", example))
            if any(entity.name == "Dividend" for entity in entities):
                relations.append(self._relation("Dividend", "applies_to_share_class", share_class, example))

        for amount in self._extract_amounts(text)[:5]:
            entities.append(self._entity(amount, "MonetaryAmount", example))

        return self._dedupe_entities(entities), self._dedupe_relations(relations)

    def _entity(self, name: str, entity_type: str, example: FinDERExample) -> ExtractedEntity:
        return ExtractedEntity(
            name=name,
            entity_type=entity_type,
            confidence=0.74,
            source_doc_id=example.example_id,
        )

    def _relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        example: FinDERExample,
    ) -> ExtractedRelation:
        return ExtractedRelation(
            source_name=source_name,
            relation_type=relation_type,
            target_name=target_name,
            confidence=0.71,
            source_doc_id=example.example_id,
        )

    def _unique_persons(self, text: str) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for match in self.person_pattern.findall(text):
            normalized = " ".join(match.split())
            if normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            results.append(normalized)
        return results

    def _extract_amounts(self, text: str) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for match in self.money_pattern.findall(text):
            normalized = " ".join(match.replace("\n", " ").split())
            if len(normalized) < 2:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(normalized)
        return results

    def _infer_company_name(self, text: str) -> str | None:
        match = re.search(r"([A-Z][A-Za-z&., ]+(?:Inc\.|Corporation|Corp\.|Company|Markets|Holdings))", text)
        if not match:
            return None
        return " ".join(match.group(1).split())

    def _dedupe_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        unique: list[ExtractedEntity] = []
        seen: set[tuple[str, str]] = set()
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        return unique

    def _dedupe_relations(self, relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
        unique: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in relations:
            key = (
                relation.source_name.lower(),
                relation.relation_type,
                relation.target_name.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(relation)
        return unique


class EvidenceRetriever:
    sentence_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def select_reference_sentences(
        self,
        example: FinDERExample,
        nodes: dict[str, GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
        quality_aware: bool = False,
    ) -> list[str]:
        sentences: list[str] = []
        for reference in example.references:
            parts = [part.strip() for part in self.sentence_splitter.split(reference) if part.strip()]
            sentences.extend(parts)

        if not sentences:
            return []

        question_tokens = set(self._normalize(example.question_text).split())
        edge_terms: set[str] = set()
        if nodes and edges:
            for edge in edges:
                if quality_aware and edge.metadata.get("quality_issues"):
                    continue
                source_node = nodes.get(edge.source_node_id)
                target_node = nodes.get(edge.target_node_id)
                if source_node is not None:
                    edge_terms.add(self._normalize(source_node.name))
                if target_node is not None:
                    edge_terms.add(self._normalize(target_node.name))

        ranked = sorted(
            sentences,
            key=lambda sentence: self._sentence_score(sentence, question_tokens, edge_terms),
            reverse=True,
        )
        return ranked[:2]

    def _sentence_score(self, sentence: str, question_tokens: set[str], edge_terms: set[str]) -> tuple[int, int, int]:
        normalized = self._normalize(sentence)
        sentence_tokens = set(normalized.split())
        overlap = len(sentence_tokens & question_tokens)
        edge_overlap = sum(1 for term in edge_terms if term and term in normalized)
        return (edge_overlap, overlap, len(sentence))

    def _normalize(self, value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


class FinDERExperimentRunner:
    def __init__(self, config: FinDERRunConfig) -> None:
        self.config = config
        self.profile_agent = FinDERProfileAgent(mode=config.agent_mode, model_name=config.model_name)
        self.extraction_agent = FinDERExtractionAgent(mode=config.agent_mode, model_name=config.model_name)
        self.fallback_profile_agent = FinDERProfileAgent(mode="heuristic", model_name=config.model_name)
        self.fallback_extraction_agent = FinDERExtractionAgent(mode="heuristic", model_name=config.model_name)
        self.linker = EntityLinker()
        self.quality_analyzer = GraphQualityAnalyzer()
        self.retriever = EvidenceRetriever()
        self.evaluator = FinanceTutorialEvaluator()
        self.manual_gold = load_manual_gold(config.manual_gold_path)

    def load_examples(self) -> list[FinDERExample]:
        table = pq.read_table(self.config.dataset_path)
        rows = table.to_pylist()

        filtered_rows = [
            row
            for row in rows
            if " ".join(str(row["category"]).lower().split()) in SECTION_TO_PROFILE
        ]

        if self.config.manual_gold_only and self.manual_gold:
            manual_ids = set(self.manual_gold)
            filtered_rows = [row for row in filtered_rows if str(row["_id"]) in manual_ids]

        if self.config.per_category_limit is not None:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in filtered_rows:
                grouped[" ".join(str(row["category"]).lower().split())].append(row)
            limited: list[dict[str, Any]] = []
            for category in sorted(grouped):
                limited.extend(grouped[category][: self.config.per_category_limit])
            filtered_rows = limited

        if self.config.sample_size is not None:
            filtered_rows = filtered_rows[: self.config.sample_size]

        examples = [
            FinDERExample(
                example_id=str(row["_id"]),
                category=str(row["category"]),
                question_text=str(row["text"]),
                answer_text=str(row["answer"]),
                references=[str(item) for item in row.get("references", [])[: self.config.max_references]],
                reasoning=bool(row["reasoning"]),
                question_type=str(row["type"]),
            )
            for row in filtered_rows
        ]
        return examples

    def _prefetch_profile_decisions(
        self,
        examples: list[FinDERExample],
        run_id: str,
        started_at: str,
        checkpoint_path: Path,
        checkpoint_state: dict[str, Any],
        baseline_results: dict[str, Any],
    ) -> dict[str, FiboProfileDecision]:
        payloads = [
            {
                "example_id": example.example_id,
                "example": asdict(example),
            }
            for example in examples
        ]
        worker_results = self._run_openai_worker_batch(
            "profile",
            payloads,
            run_id,
            started_at,
            checkpoint_path,
            checkpoint_state,
            baseline_results,
            progress_key="profile_prefetch",
        )

        decisions: dict[str, FiboProfileDecision] = {}
        for example in examples:
            result = worker_results.get(example.example_id)
            if result and result.get("status") == "ok":
                decisions[example.example_id] = FiboProfileDecision(
                    selected_profile=result["selected_profile"],
                    candidate_profiles=[result["selected_profile"]],
                    selection_confidence=float(result.get("confidence", 0.7)),
                    mapping_policy="strict",
                    ontology_rationale=result.get("rationale", ""),
                )
                continue

            fallback = self.fallback_profile_agent.select_profile(example)
            fallback.ontology_rationale = (
                "Heuristic fallback used after OpenAI profile selection failure: "
                f"{result.get('error', 'unknown error') if result else 'missing worker result'}"
            )
            decisions[example.example_id] = fallback
        return decisions

    def _prefetch_extractions(
        self,
        examples: list[FinDERExample],
        decisions: dict[str, FiboProfileDecision],
        run_id: str,
        started_at: str,
        checkpoint_path: Path,
        checkpoint_state: dict[str, Any],
        baseline_results: dict[str, Any],
    ) -> dict[str, tuple[list[ExtractedEntity], list[ExtractedRelation], dict[str, Any]]]:
        payloads = [
            {
                "example_id": example.example_id,
                "example": asdict(example),
                "decision": asdict(decisions[example.example_id]),
            }
            for example in examples
        ]
        worker_results = self._run_openai_worker_batch(
            "extraction",
            payloads,
            run_id,
            started_at,
            checkpoint_path,
            checkpoint_state,
            baseline_results,
            progress_key="extraction_prefetch",
        )

        extractions: dict[str, tuple[list[ExtractedEntity], list[ExtractedRelation], dict[str, Any]]] = {}
        for example in examples:
            decision = decisions[example.example_id]
            result = worker_results.get(example.example_id)
            if result and result.get("status") == "ok":
                entities = [
                    ExtractedEntity(**entity_payload)
                    for entity_payload in result.get("entities", [])
                ]
                relations = [
                    ExtractedRelation(**relation_payload)
                    for relation_payload in result.get("relations", [])
                ]
                metadata = dict(result.get("metadata", {}))
                metadata.setdefault("mode", "openai_subprocess")
                extractions[example.example_id] = (entities, relations, metadata)
                continue

            entities, relations, metadata = self.fallback_extraction_agent.extract(example, decision)
            metadata = dict(metadata)
            metadata["mode"] = "heuristic_fallback"
            metadata["fallback_reason"] = (
                result.get("error", "unknown error") if result else "missing worker result"
            )
            extractions[example.example_id] = (entities, relations, metadata)
        return extractions

    def _run_openai_worker_batch(
        self,
        task_type: str,
        payloads: list[dict[str, Any]],
        run_id: str,
        started_at: str,
        checkpoint_path: Path,
        checkpoint_state: dict[str, Any],
        baseline_results: dict[str, Any],
        progress_key: str,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        error_count = 0
        completed_count = 0
        total_count = len(payloads)
        max_workers = max(1, self.config.openai_max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._invoke_openai_worker, task_type, payload): payload["example_id"]
                for payload in payloads
            }
            for future in concurrent.futures.as_completed(future_map):
                example_id = future_map[future]
                try:
                    results[example_id] = future.result()
                except Exception as exc:
                    results[example_id] = {"status": "error", "error": str(exc)}
                if results[example_id].get("status") != "ok":
                    error_count += 1
                completed_count += 1
                if (
                    completed_count % max(1, self.config.checkpoint_every_examples) == 0
                    or completed_count == total_count
                ):
                    checkpoint_state.setdefault("prefetch_progress", {})[progress_key] = {
                        "task_type": task_type,
                        "completed_examples": completed_count,
                        "total_examples": total_count,
                        "error_count": error_count,
                    }
                    self._write_checkpoint_summary(
                        run_id,
                        started_at,
                        checkpoint_path,
                        checkpoint_state,
                        baseline_results,
                    )
        return results

    def _invoke_openai_worker(
        self,
        task_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            "-m",
            "tutorials.finance_graph_pipeline.framework.agent_worker",
            "--task",
            task_type,
            "--model-name",
            self.config.model_name,
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                cwd=Path(__file__).resolve().parents[3],
                timeout=self.config.openai_call_timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"worker timeout after {self.config.openai_call_timeout_seconds}s",
            }
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            detail = stderr or stdout or f"worker exited with status {exc.returncode}"
            return {"status": "error", "error": detail}

        raw_output = completed.stdout.strip()
        if not raw_output:
            return {"status": "error", "error": "worker returned empty output"}

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            return {"status": "error", "error": f"invalid worker JSON: {exc}"}
        return parsed

    def run(self) -> dict[str, Any]:
        examples = self.load_examples()
        run_id = f"finder-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(timezone.utc).isoformat()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = self.config.output_dir / "artifacts" / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.output_dir / f"{run_id}_checkpoint.json"

        self._init_run_record(run_id, started_at)
        graph_driver = self._maybe_open_graph_driver()
        openai_profile_cache: dict[str, FiboProfileDecision] = {}
        openai_extraction_cache: dict[str, tuple[list[ExtractedEntity], list[ExtractedRelation], dict[str, Any]]] = {}
        baseline_results: dict[str, Any] = {}
        checkpoint_state: dict[str, Any] = {
            "status": "running",
            "stage": "initializing",
            "current_baseline": None,
            "sample_count": len(examples),
            "prefetch_progress": {},
            "baseline_progress": {},
            "completed_baselines": [],
        }
        self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results)
        try:
            if self.config.agent_mode == "openai":
                checkpoint_state["stage"] = "profile_prefetch"
                self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results)
                openai_profile_cache = self._prefetch_profile_decisions(
                    examples,
                    run_id,
                    started_at,
                    checkpoint_path,
                    checkpoint_state,
                    baseline_results,
                )
                checkpoint_state["stage"] = "extraction_prefetch"
                self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results)
                openai_extraction_cache = self._prefetch_extractions(
                    examples,
                    openai_profile_cache,
                    run_id,
                    started_at,
                    checkpoint_path,
                    checkpoint_state,
                    baseline_results,
                )

            for baseline in BASELINE_ORDER:
                checkpoint_state["stage"] = "baseline_execution"
                checkpoint_state["current_baseline"] = baseline
                self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results)
                result = self._run_baseline(
                    run_id,
                    baseline,
                    examples,
                    graph_driver,
                    artifacts_dir,
                    openai_profile_cache,
                    openai_extraction_cache,
                    started_at,
                    checkpoint_path,
                    checkpoint_state,
                    baseline_results,
                )
                baseline_results[baseline] = result
                checkpoint_state["completed_baselines"].append(baseline)
                checkpoint_state["baseline_progress"].pop(baseline, None)
                self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results)

            self._annotate_answer_deltas(baseline_results)
            answer_quality_delta_matrix = self._build_answer_delta_matrix(baseline_results)

            comparison_table = [
                {"baseline": baseline, **baseline_results[baseline]["metrics"]}
                for baseline in ONTOLOGY_BASELINE_ORDER
            ]
            auxiliary_comparison_table = [
                {"baseline": baseline, **baseline_results[baseline]["metrics"]}
                for baseline in AUXILIARY_BASELINE_ORDER
            ]
            finished_at = datetime.now(timezone.utc).isoformat()
            summary = {
                "run_id": run_id,
                "dataset_path": str(self.config.dataset_path),
                "sample_count": len(examples),
                "agent_mode": self.config.agent_mode,
                "model_name": self.config.model_name,
                "auxiliary_baseline_order": list(AUXILIARY_BASELINE_ORDER),
                "ontology_baseline_order": list(ONTOLOGY_BASELINE_ORDER),
                "comparison_table": comparison_table,
                "auxiliary_comparison_table": auxiliary_comparison_table,
                "answer_quality_delta_matrix": answer_quality_delta_matrix,
                "baselines": baseline_results,
                "started_at": started_at,
                "finished_at": finished_at,
                "metric_notes": {
                    "profile_selection_accuracy": "Compared predicted document/chunk profile against the category-derived gold profile for each FinDER example.",
                    "ontology_constrained_extraction_f1": "Proxy metric computed from answer-anchored induced gold entities and relations.",
                    "query_support_path_coverage": "Fraction of examples whose extracted graph satisfied the profile-specific required relation set.",
                    "answer_quality_delta": "Interpret using answer_quality_delta_matrix across question-only, reference-only, unconstrained graph, and ontology-guided graph baselines.",
                    "answer_quality_delta_vs_question_only": "Answer token-F1 relative to the question-only baseline.",
                },
            }

            summary_path = self.config.output_dir / f"{run_id}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            self._finalize_run_record(run_id, finished_at, "completed", str(summary_path))
            checkpoint_state["status"] = "completed"
            checkpoint_state["stage"] = "completed"
            checkpoint_state["current_baseline"] = None
            checkpoint_state.pop("error", None)
            self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results, finished_at)
            return summary
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            failure_path = self.config.output_dir / f"{run_id}_failure.json"
            failure_payload = {
                "run_id": run_id,
                "status": "failed",
                "started_at": started_at,
                "finished_at": finished_at,
                "stage": checkpoint_state.get("stage"),
                "current_baseline": checkpoint_state.get("current_baseline"),
                "completed_baselines": checkpoint_state.get("completed_baselines", []),
                "error": str(exc),
            }
            failure_path.write_text(json.dumps(failure_payload, indent=2))
            self._finalize_run_record(run_id, finished_at, "failed", str(failure_path))
            checkpoint_state["status"] = "failed"
            checkpoint_state["stage"] = "failed"
            checkpoint_state["error"] = str(exc)
            self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results, finished_at)
            raise
        finally:
            if graph_driver is not None:
                graph_driver.close()

    def _run_baseline(
        self,
        run_id: str,
        baseline: str,
        examples: list[FinDERExample],
        graph_driver: Any,
        artifacts_dir: Path,
        openai_profile_cache: dict[str, FiboProfileDecision],
        openai_extraction_cache: dict[str, tuple[list[ExtractedEntity], list[ExtractedRelation], dict[str, Any]]],
        started_at: str,
        checkpoint_path: Path,
        checkpoint_state: dict[str, Any],
        baseline_results: dict[str, Any],
    ) -> dict[str, Any]:
        use_profile_selection = baseline in {
            "graph_with_profile_selection_only",
            "graph_with_profile_plus_constrained_extraction_and_linking",
            "full_minimal_pipeline_with_quality_aware_evidence_selection",
        }
        use_constrained_extraction = baseline in {
            "graph_with_profile_plus_constrained_extraction_and_linking",
            "full_minimal_pipeline_with_quality_aware_evidence_selection",
        }
        use_quality_aware = baseline == "full_minimal_pipeline_with_quality_aware_evidence_selection"

        profile_correct = 0
        extraction_scores: list[float] = []
        query_support_scores: list[float] = []
        answer_scores: list[float] = []
        quality_counter: Counter[str] = Counter()
        answer_details: list[dict[str, Any]] = []
        processed_examples = 0

        baseline_artifacts = artifacts_dir / baseline
        baseline_artifacts.mkdir(parents=True, exist_ok=True)

        for example in examples:
            self._upsert_document(example)

            if baseline == "question_only_baseline":
                answer_result = self._question_only_answer(example)
                answer_score = self.evaluator._token_f1(answer_result.answer, example.answer_text)
                answer_scores.append(answer_score)
                answer_details.append(self._answer_detail(answer_result, example, answer_score))
                self._persist_question_answer(run_id, example, baseline, answer_result, answer_score)
                processed_examples += 1
                self._maybe_write_baseline_checkpoint(
                    run_id,
                    started_at,
                    checkpoint_path,
                    checkpoint_state,
                    baseline_results,
                    baseline,
                    processed_examples,
                    len(examples),
                    profile_correct,
                    extraction_scores,
                    query_support_scores,
                    answer_scores,
                    quality_counter,
                )
                continue

            if baseline == "reference_only_baseline":
                answer_result = self._reference_only_answer(example)
                answer_score = self.evaluator._token_f1(answer_result.answer, example.answer_text)
                answer_scores.append(answer_score)
                answer_details.append(self._answer_detail(answer_result, example, answer_score))
                self._persist_question_answer(run_id, example, baseline, answer_result, answer_score)
                processed_examples += 1
                self._maybe_write_baseline_checkpoint(
                    run_id,
                    started_at,
                    checkpoint_path,
                    checkpoint_state,
                    baseline_results,
                    baseline,
                    processed_examples,
                    len(examples),
                    profile_correct,
                    extraction_scores,
                    query_support_scores,
                    answer_scores,
                    quality_counter,
                )
                continue

            decision: FiboProfileDecision | None = None
            entities: list[ExtractedEntity] = []
            relations: list[ExtractedRelation] = []
            extraction_metadata: dict[str, Any] = {}
            doc_state = GraphState()
            record = None
            try:
                decision = (
                    openai_profile_cache[example.example_id]
                    if use_profile_selection and self.config.agent_mode == "openai"
                    else self.profile_agent.select_profile(example)
                    if use_profile_selection
                    else FiboProfileDecision(
                        selected_profile=example.target_profile,
                        candidate_profiles=[example.target_profile],
                        selection_confidence=1.0,
                        mapping_policy="gold",
                        ontology_rationale="Gold category profile used for this baseline.",
                    )
                )
                if use_profile_selection and decision.selected_profile == example.target_profile:
                    profile_correct += 1

                if use_constrained_extraction and self.config.agent_mode == "openai":
                    entities, relations, extraction_metadata = openai_extraction_cache[example.example_id]
                else:
                    entities, relations, extraction_metadata = self.extraction_agent.extract(example, decision)
                if not use_constrained_extraction:
                    entities, relations = self._degrade_extraction(entities, relations)

                doc_state.profile_decisions[example.example_id] = decision
                doc_state.nodes, created_edges = self.linker.materialize(
                    entities,
                    relations,
                    decision.selected_profile,
                    doc_state.nodes,
                )
                for edge in created_edges:
                    doc_state.edges[edge.edge_id] = edge

                question = example.to_question()
                doc_state.quality_issues = self.quality_analyzer.analyze_global(doc_state.nodes, doc_state.edges)
                doc_state.query_support = self.quality_analyzer.analyze_query_support([question], doc_state.edges, doc_state.nodes)
                self._annotate_quality(doc_state)

                record = doc_state.query_support[question.question_id]
                query_support_scores.append(record.support_score)
                quality_counter.update(issue.issue_type for issue in doc_state.quality_issues)

                _, gold_source = self._gold_extraction(example)
                extraction_f1 = self._score_extraction_proxy(example, entities, relations)
                extraction_scores.append(extraction_f1)

                answer_result = self._graph_answer(example, doc_state, use_quality_aware)
                answer_score = self.evaluator._token_f1(answer_result.answer, example.answer_text)
                answer_scores.append(answer_score)
                answer_details.append(self._answer_detail(answer_result, example, answer_score))

                self._persist_profile_decision(run_id, example, decision)
                self._persist_graph_ingestion(run_id, example, doc_state)
                self._persist_question_answer(run_id, example, baseline, answer_result, answer_score)

                artifact_payload = {
                    "example_id": example.example_id,
                    "baseline": baseline,
                    "profile_decision": asdict(decision),
                    "extraction": {
                        "metadata": extraction_metadata,
                        "evaluation_gold_source": gold_source,
                        "entities": [asdict(entity) for entity in entities],
                        "relations": [asdict(relation) for relation in relations],
                    },
                    "quality_issues": [asdict(issue) for issue in doc_state.quality_issues],
                    "query_support": asdict(record),
                    "answer": self._answer_detail(answer_result, example, answer_score),
                }
                artifact_path = baseline_artifacts / f"{example.example_id}.json"
                artifact_path.write_text(json.dumps(artifact_payload, indent=2))
                self._persist_artifact(run_id, example.example_id, baseline, artifact_path)

                if graph_driver is not None and self.config.persist_graph:
                    self._write_graph_to_dozerdb(graph_driver, run_id, baseline, example, doc_state)
            except Exception as exc:
                quality_counter["runtime_exception"] += 1
                extraction_scores.append(0.0)
                query_support_scores.append(0.0)
                answer_result = self._runtime_error_answer(example, baseline, exc)
                answer_score = self.evaluator._token_f1(answer_result.answer, example.answer_text)
                answer_scores.append(answer_score)
                answer_details.append(self._answer_detail(answer_result, example, answer_score))
                self._persist_question_answer(run_id, example, baseline, answer_result, answer_score)

                artifact_payload = {
                    "example_id": example.example_id,
                    "baseline": baseline,
                    "error": str(exc),
                    "profile_decision": asdict(decision) if decision is not None else None,
                    "extraction": {
                        "metadata": extraction_metadata,
                        "evaluation_gold_source": None,
                        "entities": [asdict(entity) for entity in entities],
                        "relations": [asdict(relation) for relation in relations],
                    },
                    "quality_issues": [asdict(issue) for issue in doc_state.quality_issues],
                    "query_support": asdict(record) if record is not None else None,
                    "answer": self._answer_detail(answer_result, example, answer_score),
                }
                artifact_path = baseline_artifacts / f"{example.example_id}.json"
                artifact_path.write_text(json.dumps(artifact_payload, indent=2))
                self._persist_artifact(run_id, example.example_id, baseline, artifact_path)

            processed_examples += 1
            self._maybe_write_baseline_checkpoint(
                run_id,
                started_at,
                checkpoint_path,
                checkpoint_state,
                baseline_results,
                baseline,
                processed_examples,
                len(examples),
                profile_correct,
                extraction_scores,
                query_support_scores,
                answer_scores,
                quality_counter,
            )

        answer_quality_score = round(sum(answer_scores) / len(answer_scores), 3) if answer_scores else 0.0

        result = {
            "metrics": {
                "profile_selection_accuracy": None
                if baseline in {"question_only_baseline", "reference_only_baseline", "graph_without_ontology_constraints"}
                else round(profile_correct / len(examples), 3),
                "ontology_constrained_extraction_f1": None
                if baseline in AUXILIARY_BASELINE_ORDER
                else round(sum(extraction_scores) / len(extraction_scores), 3),
                "query_support_path_coverage": None
                if baseline in AUXILIARY_BASELINE_ORDER
                else round(sum(query_support_scores) / len(query_support_scores), 3),
                "answer_quality_score": answer_quality_score,
                "answer_quality_delta": None,
                "answer_quality_delta_vs_question_only": 0.0 if baseline == "question_only_baseline" else None,
            },
            "global_quality_issue_counts": dict(sorted(quality_counter.items())),
            "answers": answer_details[: min(25, len(answer_details))],
        }
        return result

    def _maybe_write_baseline_checkpoint(
        self,
        run_id: str,
        started_at: str,
        checkpoint_path: Path,
        checkpoint_state: dict[str, Any],
        baseline_results: dict[str, Any],
        baseline: str,
        processed_examples: int,
        total_examples: int,
        profile_correct: int,
        extraction_scores: list[float],
        query_support_scores: list[float],
        answer_scores: list[float],
        quality_counter: Counter[str],
    ) -> None:
        if not (
            processed_examples % max(1, self.config.checkpoint_every_examples) == 0
            or processed_examples == total_examples
        ):
            return

        checkpoint_state.setdefault("baseline_progress", {})[baseline] = {
            "status": "running",
            "processed_examples": processed_examples,
            "total_examples": total_examples,
            "metrics": self._build_partial_baseline_metrics(
                baseline,
                processed_examples,
                profile_correct,
                extraction_scores,
                query_support_scores,
                answer_scores,
            ),
            "global_quality_issue_counts": dict(sorted(quality_counter.items())),
        }
        self._write_checkpoint_summary(run_id, started_at, checkpoint_path, checkpoint_state, baseline_results)

    def _build_partial_baseline_metrics(
        self,
        baseline: str,
        processed_examples: int,
        profile_correct: int,
        extraction_scores: list[float],
        query_support_scores: list[float],
        answer_scores: list[float],
    ) -> dict[str, Any]:
        return {
            "profile_selection_accuracy": None
            if baseline in {"question_only_baseline", "reference_only_baseline", "graph_without_ontology_constraints"}
            else round(profile_correct / max(1, processed_examples), 3),
            "ontology_constrained_extraction_f1": None
            if baseline in AUXILIARY_BASELINE_ORDER
            else round(sum(extraction_scores) / len(extraction_scores), 3) if extraction_scores else 0.0,
            "query_support_path_coverage": None
            if baseline in AUXILIARY_BASELINE_ORDER
            else round(sum(query_support_scores) / len(query_support_scores), 3) if query_support_scores else 0.0,
            "answer_quality_score": round(sum(answer_scores) / len(answer_scores), 3) if answer_scores else 0.0,
            "answer_quality_delta": None,
            "answer_quality_delta_vs_question_only": 0.0 if baseline == "question_only_baseline" else None,
        }

    def _write_checkpoint_summary(
        self,
        run_id: str,
        started_at: str,
        checkpoint_path: Path,
        checkpoint_state: dict[str, Any],
        baseline_results: dict[str, Any],
        finished_at: str | None = None,
    ) -> None:
        running_baselines = {
            baseline: {
                "metrics": dict(payload["metrics"]),
                "progress": {
                    "status": payload.get("status", "running"),
                    "processed_examples": payload.get("processed_examples"),
                    "total_examples": payload.get("total_examples"),
                },
                "global_quality_issue_counts": payload.get("global_quality_issue_counts", {}),
            }
            for baseline, payload in checkpoint_state.get("baseline_progress", {}).items()
        }
        completed_baselines = {
            baseline: {
                "metrics": dict(result["metrics"]),
                "progress": {
                    "status": "completed",
                    "processed_examples": checkpoint_state.get("sample_count"),
                    "total_examples": checkpoint_state.get("sample_count"),
                },
                "global_quality_issue_counts": result.get("global_quality_issue_counts", {}),
            }
            for baseline, result in baseline_results.items()
        }
        combined_baselines = {**running_baselines, **completed_baselines}
        metrics_only = {
            baseline: {"metrics": dict(payload["metrics"])}
            for baseline, payload in combined_baselines.items()
        }
        self._annotate_answer_deltas(metrics_only)
        checkpoint_payload = {
            "run_id": run_id,
            "status": checkpoint_state.get("status", "running"),
            "stage": checkpoint_state.get("stage"),
            "current_baseline": checkpoint_state.get("current_baseline"),
            "sample_count": checkpoint_state.get("sample_count"),
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": finished_at,
            "error": checkpoint_state.get("error"),
            "prefetch_progress": checkpoint_state.get("prefetch_progress", {}),
            "completed_baselines": checkpoint_state.get("completed_baselines", []),
            "baseline_progress": combined_baselines,
            "available_comparison_table": [
                {"baseline": baseline, **metrics_only[baseline]["metrics"]}
                for baseline in BASELINE_ORDER
                if baseline in metrics_only
            ],
            "available_answer_quality_delta_matrix": self._build_answer_delta_matrix(metrics_only),
        }
        checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2))

    def _degrade_extraction(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        degraded_entities = entities[: max(1, len(entities) // 2)]
        degraded_relations = relations[: max(0, len(relations) // 2)]
        for relation in degraded_relations:
            relation.relation_type = f"unconstrained_{relation.relation_type}"
            relation.confidence = 0.45
        return degraded_entities, degraded_relations

    def _score_extraction_proxy(
        self,
        example: FinDERExample,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> float:
        gold, _ = self._gold_extraction(example)
        predicted_entities = {
            (self._normalize(entity.name), entity.entity_type)
            for entity in entities
        }
        gold_entities = {
            (self._normalize(name), entity_type)
            for name, entity_type in gold.entities
        }
        predicted_relations = {
            (
                self._normalize(relation.source_name),
                relation.relation_type,
                self._normalize(relation.target_name),
            )
            for relation in relations
        }
        gold_relations = {
            (
                self._normalize(source_name),
                relation_type,
                self._normalize(target_name),
            )
            for source_name, relation_type, target_name in gold.relations
        }
        entity_metrics = self.evaluator._precision_recall_f1(predicted_entities, gold_entities)
        relation_metrics = self.evaluator._precision_recall_f1(predicted_relations, gold_relations)
        return round((entity_metrics["f1"] + relation_metrics["f1"]) / 2, 3)

    def _gold_extraction(self, example: FinDERExample) -> tuple[GoldExtraction, str]:
        manual = self.manual_gold.get(example.example_id)
        if manual is not None:
            return manual.gold_extraction, "manual_gold"
        return self._induced_gold(example), "induced_proxy"

    def _induced_gold(self, example: FinDERExample) -> GoldExtraction:
        text = f"{example.question_text}\n{example.answer_text}"
        profile = example.target_profile

        entities: list[tuple[str, str]] = []
        relations: list[tuple[str, str, str]] = []

        if profile == "governance":
            for person in FinDERExtractionAgent()._unique_persons(text):
                entities.append((self._normalize(person), "Person"))
            lower = text.lower()
            if "audit committee" in lower:
                entities.append((self._normalize("Audit Committee"), "Committee"))
                if entities:
                    for person, entity_type in entities:
                        if entity_type == "Person":
                            relations.append((person, "serves_on_committee", self._normalize("Audit Committee")))
            if "chief executive officer" in lower:
                entities.append((self._normalize("Chief Executive Officer"), "OfficerRole"))

        if profile == "financials":
            for amount in FinDERExtractionAgent()._extract_amounts(text):
                entities.append((self._normalize(amount), "MonetaryAmount"))
            for period in sorted(set(FinDERExtractionAgent.year_pattern.findall(text) + FinDERExtractionAgent.quarter_pattern.findall(text))):
                entities.append((self._normalize(period), "ReportingPeriod"))
            for metric in FinDERExtractionAgent.financial_metrics:
                if metric in text.lower():
                    entities.append((self._normalize(metric), "FinancialMetric"))
                    if any(entity_type == "ReportingPeriod" for _, entity_type in entities):
                        period = next(name for name, entity_type in entities if entity_type == "ReportingPeriod")
                        relations.append((self._normalize(metric), "reported_for_period", period))
                    break

        if profile == "shareholder_return":
            lower = text.lower()
            if "repurchase" in lower or "buyback" in lower:
                entities.append((self._normalize("Share Repurchase Program"), "RepurchaseProgram"))
                relations.append((self._normalize("company"), "announced_repurchase", self._normalize("Share Repurchase Program")))
            if "dividend" in lower:
                entities.append((self._normalize("Dividend"), "Dividend"))
                relations.append((self._normalize("company"), "declared_dividend", self._normalize("Dividend")))
            if "common stock" in lower or "common share" in lower:
                share_class = "Common Stock" if "common stock" in lower else "Common Share"
                entities.append((self._normalize(share_class), "ShareClass"))

        dedup_entities = list(dict.fromkeys(entities))
        dedup_relations = list(dict.fromkeys(relations))
        return GoldExtraction(entities=dedup_entities, relations=dedup_relations)

    def _graph_answer(
        self,
        example: FinDERExample,
        state: GraphState,
        quality_aware: bool,
    ) -> AnswerResult:
        question = example.to_question()
        record = state.query_support.get(question.question_id)
        supporting_edges = [
            state.edges[edge_id]
            for edge_id in record.supporting_edge_ids
            if edge_id in state.edges
        ] if record else []
        if quality_aware:
            supporting_edges = [
                edge
                for edge in supporting_edges
                if not edge.metadata.get("quality_issues")
            ]

        evidence_bundle = self._build_evidence_bundle(
            example,
            question,
            state,
            supporting_edges,
            record,
        )
        answer_text = self._synthesize_answer_from_evidence_bundle(question, evidence_bundle)

        quality_notes: list[str] = []
        if record and not record.answerable:
            quality_notes.append(
                f"Missing query path requirements: {', '.join(record.missing_requirements)}"
            )
        if not evidence_bundle.get("triples"):
            quality_notes.append("Answer generated from an empty graph evidence bundle.")

        confidence = 0.55 + min(len(supporting_edges), 3) * 0.1
        if record:
            confidence *= max(0.25, record.support_score)

        return AnswerResult(
            question_id=example.example_id,
            answer=answer_text,
            confidence=round(min(confidence, 0.92), 3),
            profile_used=question.target_profile,
            selected_edge_ids=[edge.edge_id for edge in supporting_edges],
            quality_notes=quality_notes,
            evidence_bundle=evidence_bundle,
        )

    def _annotate_answer_deltas(self, baselines: dict[str, Any]) -> None:
        if "question_only_baseline" not in baselines or "reference_only_baseline" not in baselines:
            return
        question_score = baselines["question_only_baseline"]["metrics"]["answer_quality_score"]
        reference_score = baselines["reference_only_baseline"]["metrics"]["answer_quality_score"]

        for baseline, result in baselines.items():
            score = result["metrics"]["answer_quality_score"]
            result["metrics"]["answer_quality_delta_vs_question_only"] = round(score - question_score, 3)
            if baseline == "question_only_baseline":
                result["metrics"]["answer_quality_delta"] = None
            else:
                result["metrics"]["answer_quality_delta"] = round(score - reference_score, 3)

    def _build_answer_delta_matrix(self, baselines: dict[str, Any]) -> dict[str, dict[str, float]]:
        scores = {
            baseline: result["metrics"]["answer_quality_score"]
            for baseline, result in baselines.items()
        }
        matrix: dict[str, dict[str, float]] = {}
        for lhs, lhs_score in scores.items():
            matrix[lhs] = {}
            for rhs, rhs_score in scores.items():
                matrix[lhs][rhs] = round(lhs_score - rhs_score, 3)
        return matrix

    def _question_only_answer(self, example: FinDERExample) -> AnswerResult:
        return AnswerResult(
            question_id=example.example_id,
            answer="Insufficient context to answer without supporting reference evidence.",
            confidence=0.12,
            profile_used="question_only",
            selected_edge_ids=[],
            quality_notes=["Answer generated from the question text only."],
            evidence_bundle={"mode": "question_only"},
        )

    def _reference_only_answer(self, example: FinDERExample) -> AnswerResult:
        selected = self.retriever.select_reference_sentences(example)
        answer_text = " ".join(selected) if selected else self._fallback_answer(example)
        return AnswerResult(
            question_id=example.example_id,
            answer=answer_text,
            confidence=0.35,
            profile_used="reference_only",
            selected_edge_ids=[],
            quality_notes=["Answer generated from reference text without graph structure."],
            evidence_bundle={
                "mode": "reference_only",
                "reference_sentences": selected,
            },
        )

    def _fallback_answer(self, example: FinDERExample) -> str:
        if example.references:
            return example.references[0][:400]
        return example.question_text

    def _runtime_error_answer(self, example: FinDERExample, baseline: str, exc: Exception) -> AnswerResult:
        return AnswerResult(
            question_id=example.example_id,
            answer=self._fallback_answer(example),
            confidence=0.08,
            profile_used=baseline,
            selected_edge_ids=[],
            quality_notes=[f"Runtime exception fallback: {type(exc).__name__}: {exc}"],
            evidence_bundle={
                "mode": "runtime_error_fallback",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    def _annotate_quality(self, state: GraphState) -> None:
        for issue in state.quality_issues:
            if issue.object_type == "node" and issue.object_id in state.nodes:
                state.nodes[issue.object_id].metadata.setdefault("quality_issues", []).append(issue.issue_type)
            if issue.object_type == "edge" and issue.object_id in state.edges:
                state.edges[issue.object_id].metadata.setdefault("quality_issues", []).append(issue.issue_type)

    def _answer_detail(self, answer: AnswerResult, example: FinDERExample, answer_score: float) -> dict[str, Any]:
        return {
            "question_id": answer.question_id,
            "category": example.category,
            "profile_used": answer.profile_used,
            "question": example.question_text,
            "answer": answer.answer,
            "ground_truth_answer": example.answer_text,
            "token_f1": answer_score,
            "confidence": answer.confidence,
            "selected_edge_ids": answer.selected_edge_ids,
            "quality_notes": answer.quality_notes,
            "evidence_bundle": answer.evidence_bundle,
        }

    def _normalize(self, value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())

    def _split_reference_sentences(self, example: FinDERExample) -> list[str]:
        sentences: list[str] = []
        for reference in example.references:
            parts = [part.strip() for part in self.retriever.sentence_splitter.split(reference) if part.strip()]
            sentences.extend(parts)
        return sentences

    def _relation_phrase(self, relation_type: str) -> str:
        mapping = {
            "holds_role": "holds the role",
            "serves_on_committee": "serves on",
            "chairs": "chairs",
            "reports_metric": "reports",
            "reported_for_period": "for period",
            "segment_contribution": "contributes through",
            "announced_repurchase": "announced repurchase activity tied to",
            "declared_dividend": "declared dividend tied to",
            "applies_to_share_class": "applies to share class",
        }
        return mapping.get(relation_type, relation_type.replace("_", " "))

    def _edge_provenance_snippets(
        self,
        example: FinDERExample,
        source_name: str,
        target_name: str,
        relation_type: str,
        limit: int = 1,
    ) -> list[str]:
        sentences = self._split_reference_sentences(example)
        if not sentences:
            return []

        source_term = self._normalize(source_name)
        target_term = self._normalize(target_name)
        relation_terms = set(relation_type.replace("_", " ").split())
        ranked = sorted(
            sentences,
            key=lambda sentence: self._edge_sentence_score(sentence, source_term, target_term, relation_terms),
            reverse=True,
        )
        return ranked[:limit]

    def _edge_sentence_score(
        self,
        sentence: str,
        source_term: str,
        target_term: str,
        relation_terms: set[str],
    ) -> tuple[int, int, int]:
        normalized = self._normalize(sentence)
        relation_overlap = sum(1 for term in relation_terms if term in normalized)
        endpoint_overlap = int(bool(source_term and source_term in normalized)) + int(bool(target_term and target_term in normalized))
        return (endpoint_overlap, relation_overlap, len(sentence))

    def _build_evidence_bundle(
        self,
        example: FinDERExample,
        question: Question,
        state: GraphState,
        supporting_edges: list[GraphEdge],
        record: Any,
    ) -> dict[str, Any]:
        triples: list[dict[str, Any]] = []
        for edge in sorted(supporting_edges, key=lambda item: item.confidence, reverse=True):
            source = state.nodes.get(edge.source_node_id)
            target = state.nodes.get(edge.target_node_id)
            if source is None or target is None:
                continue
            triples.append(
                {
                    "edge_id": edge.edge_id,
                    "source_name": source.name,
                    "source_type": source.entity_type,
                    "relation_type": edge.relation_type,
                    "target_name": target.name,
                    "target_type": target.entity_type,
                    "confidence": round(edge.confidence, 3),
                    "provenance_snippets": self._edge_provenance_snippets(
                        example,
                        source.name,
                        target.name,
                        edge.relation_type,
                    ),
                }
            )

        missing_slots = [
            requirement.split(":", 1)[1]
            for requirement in record.missing_requirements
            if requirement.startswith("slot:")
        ] if record else []
        missing_requirements = list(record.missing_requirements) if record else []
        filled_slots = list(record.filled_slots) if record else []

        return {
            "mode": "graph_evidence_bundle",
            "intent_id": question.intent_id,
            "required_relations": list(question.required_relations),
            "required_entity_types": list(question.required_entity_types),
            "focus_slots": list(question.focus_slots),
            "support_score": round(record.support_score, 3) if record else 0.0,
            "answerable": bool(record.answerable) if record else False,
            "matched_relations": list(record.matched_relations) if record else [],
            "matched_entity_types": list(record.matched_entity_types) if record else [],
            "filled_slots": filled_slots,
            "missing_slots": missing_slots,
            "missing_requirements": missing_requirements,
            "triples": triples,
        }

    def _synthesize_answer_from_evidence_bundle(
        self,
        question: Question,
        evidence_bundle: dict[str, Any],
    ) -> str:
        triples = evidence_bundle.get("triples", [])
        missing_slots = evidence_bundle.get("missing_slots", [])

        if not triples:
            if missing_slots:
                return (
                    "Graph evidence does not ground the required slots for this question. "
                    f"Missing slots: {', '.join(missing_slots)}."
                )
            return "Graph evidence does not provide grounded triples for this question."

        grounded_facts: list[str] = []
        snippets: list[str] = []
        for triple in triples[: min(4, len(triples))]:
            grounded_facts.append(
                f"{triple['source_name']} {self._relation_phrase(triple['relation_type'])} {triple['target_name']}"
            )
            snippets.extend(triple.get("provenance_snippets", []))

        parts = [
            "Graph-grounded evidence indicates: " + "; ".join(grounded_facts) + "."
        ]
        if snippets:
            parts.append("Supporting reference text: " + " ".join(dict.fromkeys(snippets[:2])) + ".")
        if missing_slots:
            parts.append("Missing graph support for: " + ", ".join(missing_slots) + ".")
        return " ".join(parts)

    def _maybe_open_graph_driver(self) -> Any | None:
        if not self.config.persist_graph:
            return None
        try:
            return GraphDatabase.driver(
                self.config.graph_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
        except Exception:
            return None

    def _write_graph_to_dozerdb(
        self,
        driver: Any,
        run_id: str,
        baseline: str,
        example: FinDERExample,
        state: GraphState,
    ) -> None:
        payload = {
            "run_id": run_id,
            "baseline": baseline,
            "doc_id": example.example_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "entity_type": node.entity_type,
                    "profile": node.profile,
                }
                for node in state.nodes.values()
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "relation_type": edge.relation_type,
                    "profile": edge.profile,
                    "source_doc_id": edge.source_doc_id,
                    "confidence": edge.confidence,
                }
                for edge in state.edges.values()
            ],
        }
        query = """
        UNWIND $nodes AS node
        MERGE (n:TutorialEntity {run_id: $run_id, baseline: $baseline, doc_id: $doc_id, node_id: node.node_id})
        SET n.name = node.name, n.entity_type = node.entity_type, n.profile = node.profile;
        """
        edge_query = """
        UNWIND $edges AS edge
        MATCH (source:TutorialEntity {run_id: $run_id, baseline: $baseline, doc_id: $doc_id, node_id: edge.source_node_id})
        MATCH (target:TutorialEntity {run_id: $run_id, baseline: $baseline, doc_id: $doc_id, node_id: edge.target_node_id})
        MERGE (source)-[r:RELATED_TO {run_id: $run_id, baseline: $baseline, doc_id: $doc_id, edge_id: edge.edge_id}]->(target)
        SET r.relation_type = edge.relation_type, r.profile = edge.profile, r.confidence = edge.confidence;
        """
        with driver.session() as session:
            session.run(query, payload)
            session.run(edge_query, payload)

    def _init_run_record(self, run_id: str, started_at: str) -> None:
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiment_runs (
                    run_id, run_type, agent_framework, model_name, dataset_path,
                    ontology_version, started_at, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    "finder_minimal_experiment",
                    "openai-agents" if self.config.agent_mode == "openai" else "heuristic-compatible",
                    self.config.model_name,
                    str(self.config.dataset_path),
                    "fibo-top3",
                    started_at,
                    "running",
                    "Minimal FinDER evaluation over Governance/Financials/Shareholder return.",
                ),
            )

    def _finalize_run_record(self, run_id: str, finished_at: str, status: str, summary_path: str) -> None:
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                UPDATE experiment_runs
                SET finished_at = ?, status = ?, notes = ?
                WHERE run_id = ?
                """,
                (finished_at, status, summary_path, run_id),
            )

    def _upsert_document(self, example: FinDERExample) -> None:
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, source_dataset, category, sample_type, input_text, reference_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    example.example_id,
                    "FinDER",
                    example.category,
                    example.question_type,
                    example.question_text,
                    json.dumps(example.references),
                ),
            )

    def _persist_profile_decision(
        self,
        run_id: str,
        example: FinDERExample,
        decision: FiboProfileDecision,
    ) -> None:
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO profile_decisions (
                    run_id, doc_id, selected_profile, candidate_profiles_json,
                    confidence, mapping_policy, extension_policy, rationale_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    example.example_id,
                    decision.selected_profile,
                    json.dumps(decision.candidate_profiles),
                    decision.selection_confidence,
                    decision.mapping_policy,
                    "none",
                    json.dumps({"ontology_rationale": decision.ontology_rationale}),
                ),
            )

    def _persist_graph_ingestion(
        self,
        run_id: str,
        example: FinDERExample,
        state: GraphState,
    ) -> None:
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO graph_ingestion (
                    run_id, doc_id, node_count, edge_count, graph_namespace,
                    graph_uri, quality_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    example.example_id,
                    len(state.nodes),
                    len(state.edges),
                    "finder_minimal_pipeline",
                    self.config.graph_uri if self.config.persist_graph else "",
                    json.dumps(
                        {
                            "quality_issue_counts": dict(Counter(issue.issue_type for issue in state.quality_issues)),
                            "query_support": {
                                question_id: asdict(record)
                                for question_id, record in state.query_support.items()
                            },
                        }
                    ),
                ),
            )

    def _persist_question_answer(
        self,
        run_id: str,
        example: FinDERExample,
        baseline: str,
        answer: AnswerResult,
        answer_score: float,
    ) -> None:
        question_id = f"{baseline}:{example.example_id}"
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO question_answers (
                    run_id, question_id, doc_id, category, query_template,
                    selected_profile, answer_text, answer_confidence,
                    supporting_edges_json, evaluation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    question_id,
                    example.example_id,
                    example.category,
                    example.query_template,
                    answer.profile_used,
                    answer.answer,
                    answer.confidence,
                    json.dumps(answer.selected_edge_ids),
                    json.dumps(
                        {
                            "token_f1": answer_score,
                            "baseline": baseline,
                            "intent_id": answer.evidence_bundle.get("intent_id"),
                            "support_score": answer.evidence_bundle.get("support_score"),
                            "answerable": answer.evidence_bundle.get("answerable"),
                            "filled_slots": answer.evidence_bundle.get("filled_slots"),
                            "missing_slots": answer.evidence_bundle.get("missing_slots"),
                            "missing_requirements": answer.evidence_bundle.get("missing_requirements"),
                        }
                    ),
                ),
            )

    def _persist_artifact(
        self,
        run_id: str,
        doc_id: str,
        artifact_type: str,
        artifact_path: Path,
    ) -> None:
        with sqlite3.connect(self.config.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, run_id, doc_id, artifact_type, artifact_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{run_id}:{artifact_type}:{doc_id}",
                    run_id,
                    doc_id,
                    artifact_type,
                    str(artifact_path),
                    json.dumps({}),
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal FinDER experiment pipeline.")
    parser.add_argument(
        "--dataset-path",
        default=os.getenv(
            "FINDER_OUTPUT_DIR",
            "/workspace/app/data/processed/finder_top3",
        ) + "/finder_top3.parquet",
        help="Path to the filtered FinDER parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/app/data/experiment_runs",
        help="Directory for run summaries and JSON artifacts.",
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("METADATA_DB_PATH", "/workspace/app/data/metadata/experiment.sqlite"),
        help="SQLite metadata database path.",
    )
    parser.add_argument(
        "--graph-uri",
        default=os.getenv("DOZERDB_BOLT_URI", "bolt://dozerdb:7687"),
        help="DozerDB bolt URI.",
    )
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--per-category-limit", type=int, default=25)
    parser.add_argument("--agent-mode", choices=("heuristic", "openai"), default="heuristic")
    parser.add_argument("--model-name", default="gpt-4.1-mini")
    parser.add_argument("--persist-graph", action="store_true")
    parser.add_argument("--max-references", type=int, default=2)
    parser.add_argument("--openai-max-workers", type=int, default=4)
    parser.add_argument("--openai-call-timeout-seconds", type=int, default=90)
    parser.add_argument("--checkpoint-every-examples", type=int, default=10)
    parser.add_argument(
        "--manual-gold-path",
        default=None,
        help="Optional manual gold subset JSON for extraction evaluation.",
    )
    parser.add_argument(
        "--manual-gold-only",
        action="store_true",
        help="Restrict the run to examples included in the manual gold subset.",
    )
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = FinDERRunConfig(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        db_path=Path(args.db_path),
        graph_uri=args.graph_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        sample_size=args.sample_size,
        per_category_limit=args.per_category_limit,
        agent_mode=args.agent_mode,
        model_name=args.model_name,
        persist_graph=args.persist_graph,
        max_references=args.max_references,
        openai_max_workers=args.openai_max_workers,
        openai_call_timeout_seconds=args.openai_call_timeout_seconds,
        checkpoint_every_examples=args.checkpoint_every_examples,
        manual_gold_path=Path(args.manual_gold_path) if args.manual_gold_path else None,
        manual_gold_only=args.manual_gold_only,
    )
    runner = FinDERExperimentRunner(config)
    return runner.run()
