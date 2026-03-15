from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .agents import AnswerGenerator, EntityLinker, EvidenceSelector, ExtractionAgent, FiboProfileAgent
from .fibo_profiles import SECTION_TO_PROFILE
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

AUXILIARY_BASELINE_ORDER = (
    "question_only_baseline",
    "reference_only_baseline",
)

ONTOLOGY_BASELINE_ORDER = (
    "graph_without_ontology_constraints",
    "graph_with_profile_selection_only",
    "graph_with_profile_plus_constrained_extraction_and_linking",
    "full_minimal_pipeline_with_quality_aware_evidence_selection",
)

BASELINE_ORDER = AUXILIARY_BASELINE_ORDER + ONTOLOGY_BASELINE_ORDER
PRIMARY_ANSWER_BASELINE = "reference_only_baseline"


@dataclass(frozen=True)
class GoldExtraction:
    entities: list[tuple[str, str]]
    relations: list[tuple[str, str, str]]


class UnconstrainedExtractionAgent:
    person_pattern = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
    money_pattern = re.compile(r"(USD [0-9]+(?:\.[0-9]+)?(?: billion| million)?)")
    quarter_pattern = re.compile(r"\b(Q[1-4] 20[0-9]{2})\b")
    stop_person_tokens = {"Executive", "Board", "Committee", "Return", "Banking"}

    def extract(self, document: Document) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        lowered = document.text.lower()

        if "northwind bank" in lowered:
            entities.append(self._entity("Northwind Bank", "Organization", document))
        if "board" in lowered:
            entities.append(self._entity("Board", "GovernanceGroup", document))
        if "audit committee" in lowered:
            entities.append(self._entity("Audit Committee", "Committee", document))

        for person in self.person_pattern.findall(document.text):
            first, second = person.split(" ", 1)
            if first in self.stop_person_tokens or second in self.stop_person_tokens:
                continue
            entities.append(self._entity(person, "Person", document))

        if "chief executive officer" in lowered:
            entities.append(self._entity("Chief Executive Officer", "RoleMention", document))
            relations.append(self._relation("Jane Doe", "mentions_role", "Chief Executive Officer", document))
        if "audit committee includes" in lowered:
            for member in ("John Smith", "Mira Patel"):
                relations.append(self._relation(member, "mentions_membership", "Audit Committee", document))

        if "net income" in lowered:
            entities.append(self._entity("Net Income", "Metric", document))
        if "consumer banking" in lowered:
            entities.append(self._entity("Consumer Banking", "BusinessSegment", document))
        for period in self.quarter_pattern.findall(document.text):
            entities.append(self._entity(period, "TimePeriod", document))
        for amount in self.money_pattern.findall(document.text):
            entities.append(self._entity(amount, "Value", document))
        if "reported net income" in lowered:
            relations.append(self._relation("Northwind Bank", "mentions_metric", "Net Income", document))
            relations.append(self._relation("Net Income", "mentions_period", "Q4 2025", document))
        if "consumer banking contributed" in lowered:
            relations.append(self._relation("Consumer Banking", "mentions_value", "USD 900 million", document))

        if "share repurchase program" in lowered:
            entities.append(self._entity("Share Repurchase Program", "CorporateAction", document))
            relations.append(
                self._relation("Northwind Bank", "mentions_capital_return", "Share Repurchase Program", document)
            )
        if "cash dividend" in lowered:
            entities.append(self._entity("Quarterly Cash Dividend", "CorporateAction", document))
            relations.append(
                self._relation("Northwind Bank", "mentions_capital_return", "Quarterly Cash Dividend", document)
            )
        if "common share" in lowered:
            entities.append(self._entity("Common Share", "SecurityClass", document))
            relations.append(self._relation("Quarterly Cash Dividend", "mentions_security", "Common Share", document))

        return self._deduplicate_entities(entities), relations

    def _deduplicate_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        seen: set[tuple[str, str, str]] = set()
        unique: list[ExtractedEntity] = []
        for entity in entities:
            key = (entity.source_doc_id, entity.name, entity.entity_type)
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        return unique

    def _entity(self, name: str, entity_type: str, document: Document) -> ExtractedEntity:
        return ExtractedEntity(
            name=name,
            entity_type=entity_type,
            confidence=0.58,
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
            confidence=0.52,
            source_doc_id=document.doc_id,
        )


class FinanceTutorialEvaluator:
    def __init__(self) -> None:
        self.profile_agent = FiboProfileAgent()
        self.constrained_extractor = ExtractionAgent()
        self.unconstrained_extractor = UnconstrainedExtractionAgent()
        self.linker = EntityLinker()
        self.quality_analyzer = GraphQualityAnalyzer()
        self.selector = EvidenceSelector()
        self.answer_generator = AnswerGenerator()

    def load_dataset(
        self, dataset_path: str | Path
    ) -> tuple[list[Document], list[Question], dict[str, str], dict[str, GoldExtraction]]:
        payload = json.loads(Path(dataset_path).read_text())
        documents = [Document(**item) for item in payload["documents"]]
        questions = [Question(**item) for item in payload["questions"]]
        gold_profiles = dict(payload["gold_profiles"])

        gold_extractions: dict[str, GoldExtraction] = {}
        for doc_id, annotation in payload["gold_extractions"].items():
            entities = [(item["name"], item["entity_type"]) for item in annotation["entities"]]
            relations = [
                (item["source_name"], item["relation_type"], item["target_name"])
                for item in annotation["relations"]
            ]
            gold_extractions[doc_id] = GoldExtraction(entities=entities, relations=relations)

        return documents, questions, gold_profiles, gold_extractions

    def evaluate(self, dataset_path: str | Path) -> dict[str, object]:
        documents, questions, gold_profiles, gold_extractions = self.load_dataset(dataset_path)

        results: dict[str, object] = {
            "dataset_path": str(Path(dataset_path)),
            "baseline_order": list(BASELINE_ORDER),
            "auxiliary_baseline_order": list(AUXILIARY_BASELINE_ORDER),
            "ontology_baseline_order": list(ONTOLOGY_BASELINE_ORDER),
            "baselines": {},
        }

        for baseline in BASELINE_ORDER:
            baseline_result = self._run_baseline(baseline, documents, questions, gold_profiles, gold_extractions)
            results["baselines"][baseline] = baseline_result

        self._annotate_answer_deltas(results["baselines"])
        results["answer_quality_delta_matrix"] = self._build_answer_delta_matrix(results["baselines"])

        results["comparison_table"] = [
            {"baseline": baseline, **results["baselines"][baseline]["metrics"]}
            for baseline in ONTOLOGY_BASELINE_ORDER
        ]
        results["auxiliary_comparison_table"] = [
            {"baseline": baseline, **results["baselines"][baseline]["metrics"]}
            for baseline in AUXILIARY_BASELINE_ORDER
        ]

        return results

    def _run_baseline(
        self,
        baseline: str,
        documents: list[Document],
        questions: list[Question],
        gold_profiles: dict[str, str],
        gold_extractions: dict[str, GoldExtraction],
    ) -> dict[str, object]:
        if baseline == "question_only_baseline":
            answers = self._answer_question_only(questions)
            answer_score, answer_details = self._score_answers(answers, questions)
            return {
                "metrics": {
                    "profile_selection_accuracy": None,
                    "ontology_constrained_extraction_f1": None,
                    "query_support_path_coverage": None,
                    "answer_quality_score": answer_score,
                    "answer_quality_delta": None,
                    "answer_quality_delta_vs_question_only": 0.0,
                },
                "supporting_metrics": {},
                "profile_decisions": {},
                "global_quality_issue_counts": {},
                "query_support": {},
                "answers": answer_details,
            }

        if baseline == "reference_only_baseline":
            answers = self._answer_reference_only(documents, questions)
            answer_score, answer_details = self._score_answers(answers, questions)
            return {
                "metrics": {
                    "profile_selection_accuracy": None,
                    "ontology_constrained_extraction_f1": None,
                    "query_support_path_coverage": None,
                    "answer_quality_score": answer_score,
                    "answer_quality_delta": None,
                    "answer_quality_delta_vs_question_only": None,
                },
                "supporting_metrics": {},
                "profile_decisions": {},
                "global_quality_issue_counts": {},
                "query_support": {},
                "answers": answer_details,
            }

        use_profile_selection = baseline in {
            "graph_with_profile_selection_only",
            "graph_with_profile_plus_constrained_extraction_and_linking",
            "full_minimal_pipeline_with_quality_aware_evidence_selection",
        }
        use_constrained_extraction = baseline in {
            "graph_with_profile_plus_constrained_extraction_and_linking",
            "full_minimal_pipeline_with_quality_aware_evidence_selection",
        }
        use_linking = use_constrained_extraction
        use_quality_aware_selection = baseline == "full_minimal_pipeline_with_quality_aware_evidence_selection"

        state = GraphState()
        extracted_entities_by_doc: dict[str, list[ExtractedEntity]] = {}
        extracted_relations_by_doc: dict[str, list[ExtractedRelation]] = {}

        for document in documents:
            if use_profile_selection:
                decision = self.profile_agent.select_profile(document)
                selected_profile = decision.selected_profile
                state.profile_decisions[document.doc_id] = decision
            else:
                selected_profile = gold_profiles[document.doc_id]

            if use_constrained_extraction:
                decision = state.profile_decisions.get(document.doc_id) or FiboProfileDecision(
                    selected_profile=selected_profile,
                    candidate_profiles=[selected_profile],
                    selection_confidence=1.0,
                    mapping_policy="gold",
                    ontology_rationale="Gold profile used for constrained extraction baseline.",
                )
                entities, relations = self.constrained_extractor.extract(document, decision)
            else:
                entities, relations = self.unconstrained_extractor.extract(document)

            extracted_entities_by_doc[document.doc_id] = entities
            extracted_relations_by_doc[document.doc_id] = relations

            if use_linking:
                state.nodes, created_edges = self.linker.materialize(
                    entities,
                    relations,
                    selected_profile,
                    state.nodes,
                )
            else:
                state.nodes, created_edges = self._materialize_without_linking(
                    entities,
                    relations,
                    selected_profile,
                    state.nodes,
                )

            for edge in created_edges:
                state.edges[edge.edge_id] = edge

        state.quality_issues = self.quality_analyzer.analyze_global(state.nodes, state.edges)
        state.query_support = self.quality_analyzer.analyze_query_support(questions, state.edges, state.nodes)

        if use_quality_aware_selection:
            self._annotate_quality_metadata(state)

        answers = [
            self._answer_from_graph(question, state, quality_aware=use_quality_aware_selection)
            for question in questions
        ]
        answer_score, answer_details = self._score_answers(answers, questions)

        extraction_metrics = self._score_extractions(extracted_entities_by_doc, extracted_relations_by_doc, gold_extractions)
        profile_accuracy = self._score_profiles(state.profile_decisions, gold_profiles) if use_profile_selection else None

        quality_counts = dict(sorted(Counter(issue.issue_type for issue in state.quality_issues).items()))
        query_support_path_coverage = round(
            sum(record.support_score for record in state.query_support.values()) / len(questions),
            3,
        )

        return {
            "metrics": {
                "profile_selection_accuracy": profile_accuracy,
                "ontology_constrained_extraction_f1": extraction_metrics["ontology_constrained_extraction_f1"],
                "query_support_path_coverage": query_support_path_coverage,
                "answer_quality_score": answer_score,
                "answer_quality_delta": None,
                "answer_quality_delta_vs_question_only": None,
            },
            "supporting_metrics": extraction_metrics,
            "profile_decisions": {
                doc_id: {
                    "selected_profile": decision.selected_profile,
                    "selection_confidence": decision.selection_confidence,
                    "mapping_policy": decision.mapping_policy,
                }
                for doc_id, decision in state.profile_decisions.items()
            },
            "global_quality_issue_counts": quality_counts,
            "query_support": {
                question_id: {
                    "support_score": round(record.support_score, 3),
                    "answerable": record.answerable,
                    "missing_requirements": record.missing_requirements,
                }
                for question_id, record in state.query_support.items()
            },
            "answers": answer_details,
        }

    def _annotate_answer_deltas(self, baselines: dict[str, object]) -> None:
        question_score = baselines["question_only_baseline"]["metrics"]["answer_quality_score"]
        reference_score = baselines[PRIMARY_ANSWER_BASELINE]["metrics"]["answer_quality_score"]

        for baseline, result in baselines.items():
            score = result["metrics"]["answer_quality_score"]
            result["metrics"]["answer_quality_delta_vs_question_only"] = round(score - question_score, 3)
            if baseline == "question_only_baseline":
                result["metrics"]["answer_quality_delta"] = None
            else:
                result["metrics"]["answer_quality_delta"] = round(score - reference_score, 3)

    def _build_answer_delta_matrix(self, baselines: dict[str, object]) -> dict[str, dict[str, float]]:
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

    def _score_profiles(
        self,
        profile_decisions: dict[str, FiboProfileDecision],
        gold_profiles: dict[str, str],
    ) -> float:
        correct = sum(
            1 for doc_id, gold_profile in gold_profiles.items()
            if profile_decisions.get(doc_id) and profile_decisions[doc_id].selected_profile == gold_profile
        )
        return round(correct / len(gold_profiles), 3)

    def _score_extractions(
        self,
        predicted_entities: dict[str, list[ExtractedEntity]],
        predicted_relations: dict[str, list[ExtractedRelation]],
        gold_extractions: dict[str, GoldExtraction],
    ) -> dict[str, float]:
        predicted_entity_set = {
            (doc_id, self._normalize_text(entity.name), entity.entity_type)
            for doc_id, entities in predicted_entities.items()
            for entity in entities
        }
        gold_entity_set = {
            (doc_id, self._normalize_text(name), entity_type)
            for doc_id, gold in gold_extractions.items()
            for name, entity_type in gold.entities
        }

        predicted_relation_set = {
            (
                doc_id,
                self._normalize_text(relation.source_name),
                relation.relation_type,
                self._normalize_text(relation.target_name),
            )
            for doc_id, relations in predicted_relations.items()
            for relation in relations
        }
        gold_relation_set = {
            (
                doc_id,
                self._normalize_text(source_name),
                relation_type,
                self._normalize_text(target_name),
            )
            for doc_id, gold in gold_extractions.items()
            for source_name, relation_type, target_name in gold.relations
        }

        entity_metrics = self._precision_recall_f1(predicted_entity_set, gold_entity_set)
        relation_metrics = self._precision_recall_f1(predicted_relation_set, gold_relation_set)
        ontology_f1 = round((entity_metrics["f1"] + relation_metrics["f1"]) / 2, 3)

        return {
            "entity_precision": entity_metrics["precision"],
            "entity_recall": entity_metrics["recall"],
            "entity_f1": entity_metrics["f1"],
            "relation_precision": relation_metrics["precision"],
            "relation_recall": relation_metrics["recall"],
            "relation_f1": relation_metrics["f1"],
            "ontology_constrained_extraction_f1": ontology_f1,
        }

    def _precision_recall_f1(
        self,
        predicted: set[tuple[object, ...]],
        gold: set[tuple[object, ...]],
    ) -> dict[str, float]:
        if not predicted and not gold:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        true_positive = len(predicted & gold)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(gold) if gold else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    def _answer_question_only(self, questions: list[Question]) -> list[AnswerResult]:
        answers: list[AnswerResult] = []
        for question in questions:
            answers.append(
                AnswerResult(
                    question_id=question.question_id,
                    answer="Insufficient context to answer without supporting reference evidence.",
                    confidence=0.12,
                    profile_used="question_only",
                    selected_edge_ids=[],
                    quality_notes=["Answer generated from the question text only."],
                )
            )
        return answers

    def _answer_reference_only(self, documents: list[Document], questions: list[Question]) -> list[AnswerResult]:
        documents_by_profile = {
            SECTION_TO_PROFILE[document.section_type.lower()]: document
            for document in documents
        }
        answers: list[AnswerResult] = []
        for question in questions:
            document = documents_by_profile[question.target_profile]
            answer = self._first_sentence(document.text)
            answers.append(
                AnswerResult(
                    question_id=question.question_id,
                    answer=answer,
                    confidence=0.45,
                    profile_used="reference_only",
                    selected_edge_ids=[],
                    quality_notes=["Answer generated from reference text without graph structure."],
                )
            )
        return answers

    def _answer_from_graph(
        self,
        question: Question,
        state: GraphState,
        quality_aware: bool,
    ) -> AnswerResult:
        query_support = state.query_support.get(question.question_id)
        selected_edges = self.selector.select_edges(question, state.edges)

        if quality_aware:
            selected_edges = [
                edge
                for edge in selected_edges
                if not edge.metadata.get("quality_issues")
            ]

        quality_notes: list[str] = []
        if query_support and not query_support.answerable:
            quality_notes.append(
                f"Missing query path requirements: {', '.join(query_support.missing_requirements)}"
            )
        if not selected_edges:
            quality_notes.append("No high-quality supporting edges were selected.")

        answer, confidence = self.answer_generator.generate(question, selected_edges, state.nodes)
        if quality_aware and query_support:
            confidence *= max(0.25, query_support.support_score)

        return AnswerResult(
            question_id=question.question_id,
            answer=answer,
            confidence=round(confidence, 3),
            profile_used=question.target_profile,
            selected_edge_ids=[edge.edge_id for edge in selected_edges],
            quality_notes=quality_notes,
        )

    def _materialize_without_linking(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        profile: str,
        existing_nodes: dict[str, GraphNode],
    ) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
        nodes = existing_nodes
        created_edges: list[GraphEdge] = []

        for entity in entities:
            node_id = self._raw_node_id(entity.source_doc_id, entity.name, entity.entity_type)
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
            node.metadata["last_entity_confidence"] = entity.confidence

        for index, relation in enumerate(relations, start=1):
            source_id = self._ensure_raw_node(nodes, relation.source_doc_id, relation.source_name, profile)
            target_id = self._ensure_raw_node(nodes, relation.source_doc_id, relation.target_name, profile)
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
                    metadata={"extraction_confidence": relation.confidence, "linking": "disabled"},
                )
            )

        return nodes, created_edges

    def _ensure_raw_node(
        self,
        nodes: dict[str, GraphNode],
        doc_id: str,
        name: str,
        profile: str,
    ) -> str:
        node_id = self._raw_node_id(doc_id, name, "UnresolvedEntity")
        if node_id not in nodes:
            nodes[node_id] = GraphNode(
                node_id=node_id,
                name=name,
                entity_type="UnresolvedEntity",
                profile=profile,
            )
        nodes[node_id].aliases.add(name)
        nodes[node_id].source_doc_ids.add(doc_id)
        return node_id

    def _raw_node_id(self, doc_id: str, name: str, entity_type: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        entity_slug = re.sub(r"[^a-z0-9]+", "_", entity_type.lower()).strip("_")
        return f"{doc_id}:{slug}:{entity_slug}"

    def _annotate_quality_metadata(self, state: GraphState) -> None:
        for issue in state.quality_issues:
            if issue.object_type == "node" and issue.object_id in state.nodes:
                state.nodes[issue.object_id].metadata.setdefault("quality_issues", []).append(issue.issue_type)
            if issue.object_type == "edge" and issue.object_id in state.edges:
                state.edges[issue.object_id].metadata.setdefault("quality_issues", []).append(issue.issue_type)

        for record in state.query_support.values():
            for edge_id in record.supporting_edge_ids:
                if edge_id in state.edges:
                    state.edges[edge_id].metadata.setdefault("query_support", []).append(record.question_id)

    def _score_answers(
        self,
        answers: list[AnswerResult],
        questions: list[Question],
    ) -> tuple[float, list[dict[str, object]]]:
        question_map = {question.question_id: question for question in questions}
        details: list[dict[str, object]] = []
        scores: list[float] = []

        for answer in answers:
            question = question_map[answer.question_id]
            token_f1 = self._token_f1(answer.answer, question.ground_truth_answer)
            scores.append(token_f1)
            details.append(
                {
                    "question_id": answer.question_id,
                    "profile_used": answer.profile_used,
                    "answer": answer.answer,
                    "ground_truth_answer": question.ground_truth_answer,
                    "token_f1": token_f1,
                    "confidence": answer.confidence,
                    "selected_edge_ids": answer.selected_edge_ids,
                    "quality_notes": answer.quality_notes,
                }
            )

        overall = round(sum(scores) / len(scores), 3)
        return overall, details

    def _token_f1(self, predicted: str, gold: str) -> float:
        predicted_tokens = self._normalize_text(predicted).split()
        gold_tokens = self._normalize_text(gold).split()
        if not predicted_tokens or not gold_tokens:
            return 0.0

        predicted_counter = Counter(predicted_tokens)
        gold_counter = Counter(gold_tokens)
        overlap = sum(min(predicted_counter[token], gold_counter[token]) for token in predicted_counter)
        if overlap == 0:
            return 0.0

        precision = overlap / len(predicted_tokens)
        recall = overlap / len(gold_tokens)
        return round(2 * precision * recall / (precision + recall), 3)

    def _first_sentence(self, text: str) -> str:
        sentence = re.split(r"(?<=\w)\.\s+(?=[A-Z])", text, maxsplit=1)[0].strip()
        if sentence and not sentence.endswith("."):
            sentence += "."
        return sentence

    def _normalize_text(self, value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())
