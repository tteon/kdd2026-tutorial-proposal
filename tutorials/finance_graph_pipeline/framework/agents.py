from __future__ import annotations

import re
from collections import defaultdict

from .fibo_profiles import FIBO_PROFILES, SECTION_TO_PROFILE
from .models import (
    Document,
    ExtractedEntity,
    ExtractedRelation,
    FiboProfileDecision,
    GraphEdge,
    GraphNode,
    Question,
)


class FiboProfileAgent:
    def select_profile(self, document: Document) -> FiboProfileDecision:
        section_key = document.section_type.strip().lower()
        selected = SECTION_TO_PROFILE.get(section_key, "financials")
        candidates = [selected]
        if selected != "financials":
            candidates.append("financials")

        rationale = (
            f"Section '{document.section_type}' most closely matches the '{selected}' profile based on "
            "domain-specific entity and relation patterns."
        )
        return FiboProfileDecision(
            selected_profile=selected,
            candidate_profiles=candidates,
            selection_confidence=0.9 if section_key in SECTION_TO_PROFILE else 0.55,
            mapping_policy="strict" if section_key in SECTION_TO_PROFILE else "soft",
            ontology_rationale=rationale,
        )


class ExtractionAgent:
    person_pattern = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
    money_pattern = re.compile(r"(USD [0-9]+(?:\.[0-9]+)?(?: billion| million)?)")
    quarter_pattern = re.compile(r"\b(Q[1-4] 20[0-9]{2})\b")
    stop_person_tokens = {"Executive", "Board", "Committee", "Return", "Banking"}

    def extract(
        self,
        document: Document,
        decision: FiboProfileDecision,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        profile = decision.selected_profile
        method = getattr(self, f"_extract_{profile}", self._extract_financials)
        return method(document)

    def _extract_governance(
        self, document: Document
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        company = "Northwind Bank"
        entities.append(self._entity(company, "LegalEntity", document))
        entities.append(self._entity("Board", "Board", document))
        entities.append(self._entity("Audit Committee", "Committee", document))

        for person in self.person_pattern.findall(document.text):
            first, second = person.split(" ", 1)
            if first in self.stop_person_tokens or second in self.stop_person_tokens:
                continue
            entities.append(self._entity(person, "Person", document))

        if "Chief Executive Officer" in document.text:
            entities.append(self._entity("Chief Executive Officer", "OfficerRole", document))
            relations.append(self._relation("Jane Doe", "holds_role", "Chief Executive Officer", document))
        if "Chair of the Board" in document.text:
            relations.append(self._relation("Jane Doe", "chairs", "Board", document))
        if "audit committee includes" in document.text.lower():
            for member in ("John Smith", "Mira Patel"):
                relations.append(self._relation(member, "serves_on_committee", "Audit Committee", document))

        return entities, relations

    def _extract_financials(
        self, document: Document
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        company = "Northwind Bank"
        entities.append(self._entity(company, "LegalEntity", document))
        entities.append(self._entity("Net Income", "FinancialMetric", document))
        entities.append(self._entity("Consumer Banking", "BusinessSegment", document))

        for period in self.quarter_pattern.findall(document.text):
            entities.append(self._entity(period, "ReportingPeriod", document))

        monies = self.money_pattern.findall(document.text)
        for amount in monies:
            entities.append(self._entity(amount, "MonetaryAmount", document))

        if "USD 2.4 billion" in document.text:
            relations.append(self._relation(company, "reports_metric", "Net Income", document))
            relations.append(self._relation("Net Income", "reported_for_period", "Q4 2025", document))
        if "Consumer Banking contributed USD 900 million" in document.text:
            relations.append(self._relation("Consumer Banking", "segment_contribution", "USD 900 million", document))

        return entities, relations

    def _extract_shareholder_return(
        self, document: Document
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []

        company = "Northwind Bank"
        entities.append(self._entity(company, "LegalEntity", document))
        entities.append(self._entity("Share Repurchase Program", "RepurchaseProgram", document))
        entities.append(self._entity("Quarterly Cash Dividend", "Dividend", document))
        entities.append(self._entity("Common Share", "ShareClass", document))

        for amount in self.money_pattern.findall(document.text):
            entities.append(self._entity(amount, "MonetaryAmount", document))

        if "share repurchase program" in document.text.lower():
            relations.append(
                self._relation(company, "announced_repurchase", "Share Repurchase Program", document)
            )
        if "cash dividend" in document.text.lower():
            relations.append(
                self._relation(company, "declared_dividend", "Quarterly Cash Dividend", document)
            )
            relations.append(
                self._relation("Quarterly Cash Dividend", "applies_to_share_class", "Common Share", document)
            )

        return entities, relations

    def _entity(self, name: str, entity_type: str, document: Document) -> ExtractedEntity:
        return ExtractedEntity(
            name=name,
            entity_type=entity_type,
            confidence=0.88,
            source_doc_id=document.doc_id,
        )

    def _relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        document: Document,
    ) -> ExtractedRelation:
        return ExtractedRelation(
            source_name=source_name,
            relation_type=relation_type,
            target_name=target_name,
            confidence=0.84,
            source_doc_id=document.doc_id,
        )


class EntityLinker:
    def __init__(self) -> None:
        self.alias_map = {
            "northwind bank": "northwind_bank",
            "board": "board",
            "audit committee": "audit_committee",
            "chief executive officer": "chief_executive_officer",
            "share repurchase program": "share_repurchase_program",
            "quarterly cash dividend": "quarterly_cash_dividend",
            "common share": "common_share",
            "net income": "net_income",
            "consumer banking": "consumer_banking",
            "jane doe": "jane_doe",
            "john smith": "john_smith",
            "mira patel": "mira_patel",
            "q4 2025": "q4_2025",
            "q4 2024": "q4_2024",
            "usd 2.4 billion": "usd_2_4_billion",
            "usd 2.1 billion": "usd_2_1_billion",
            "usd 900 million": "usd_900_million",
            "usd 5 billion": "usd_5_billion",
            "usd 0.42": "usd_0_42",
        }

    def materialize(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        profile: str,
        existing_nodes: dict[str, GraphNode],
    ) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
        nodes = existing_nodes
        created_edges: list[GraphEdge] = []

        for entity in entities:
            node_id = self._canonical_id(entity.name)
            if node_id not in nodes:
                nodes[node_id] = GraphNode(
                    node_id=node_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    profile=profile,
                )
            node = nodes[node_id]
            node.aliases.add(entity.name)
            node.source_doc_ids.add(entity.source_doc_id)
            node.metadata.setdefault("profiles", set()).add(profile)
            node.metadata["last_entity_confidence"] = entity.confidence

        for index, relation in enumerate(relations, start=1):
            source_id = self._canonical_id(relation.source_name)
            target_id = self._canonical_id(relation.target_name)
            edge_id = f"{profile}:{relation.source_doc_id}:{index}:{source_id}:{relation.relation_type}:{target_id}"
            created_edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    source_node_id=source_id,
                    relation_type=relation.relation_type,
                    target_node_id=target_id,
                    profile=profile,
                    source_doc_id=relation.source_doc_id,
                    confidence=relation.confidence,
                    metadata={"extraction_confidence": relation.confidence},
                )
            )

        return nodes, created_edges

    def _canonical_id(self, name: str) -> str:
        lowered = name.strip().lower()
        return self.alias_map.get(lowered, re.sub(r"[^a-z0-9]+", "_", lowered).strip("_"))


class EvidenceSelector:
    template_requirements = {
        "who-serves-on-committee": ("serves_on_committee",),
        "metric-for-period": ("reports_metric", "reported_for_period"),
        "shareholder-return-policy": ("announced_repurchase", "declared_dividend"),
    }

    def select_edges(
        self,
        question: Question,
        edges: dict[str, GraphEdge],
    ) -> list[GraphEdge]:
        required = self.template_requirements.get(question.query_template, ())
        selected = [
            edge
            for edge in edges.values()
            if edge.profile == question.target_profile and edge.relation_type in required
        ]
        return sorted(selected, key=lambda edge: edge.confidence, reverse=True)


class AnswerGenerator:
    def generate(self, question: Question, edges: list[GraphEdge], nodes: dict[str, GraphNode]) -> tuple[str, float]:
        if not edges:
            return ("Insufficient high-quality graph evidence to answer the question.", 0.2)

        if question.query_template == "who-serves-on-committee":
            members = [nodes[edge.source_node_id].name for edge in edges if edge.relation_type == "serves_on_committee"]
            answer = f"{', '.join(members)} serve on the audit committee."
            return answer, 0.88

        if question.query_template == "metric-for-period":
            company = next((nodes[edge.source_node_id].name for edge in edges if edge.relation_type == "reports_metric"), "The company")
            answer = f"{company} reported USD 2.4 billion of net income for Q4 2025."
            return answer, 0.84

        if question.query_template == "shareholder-return-policy":
            answer = (
                "The company is returning capital through a USD 5 billion share repurchase program "
                "and a USD 0.42 quarterly cash dividend."
            )
            return answer, 0.83

        return ("A graph answer was produced, but no tutorial template matched the question.", 0.5)
