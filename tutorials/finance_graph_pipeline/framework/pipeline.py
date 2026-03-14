from __future__ import annotations

import json
from pathlib import Path

from .agents import (
    AnswerGenerator,
    EntityLinker,
    EvidenceSelector,
    ExtractionAgent,
    FiboProfileAgent,
)
from .models import AnswerResult, Document, GraphState, Question
from .quality import GraphQualityAnalyzer


class FinanceTutorialPipeline:
    def __init__(self) -> None:
        self.profile_agent = FiboProfileAgent()
        self.extraction_agent = ExtractionAgent()
        self.linker = EntityLinker()
        self.quality_analyzer = GraphQualityAnalyzer()
        self.selector = EvidenceSelector()
        self.answer_generator = AnswerGenerator()
        self.state = GraphState()

    def load_dataset(self, dataset_path: str | Path) -> tuple[list[Document], list[Question]]:
        payload = json.loads(Path(dataset_path).read_text())
        documents = [Document(**item) for item in payload["documents"]]
        questions = [Question(**item) for item in payload["questions"]]
        return documents, questions

    def index_documents(self, documents: list[Document]) -> GraphState:
        for document in documents:
            decision = self.profile_agent.select_profile(document)
            entities, relations = self.extraction_agent.extract(document, decision)
            self.state.profile_decisions[document.doc_id] = decision
            self.state.nodes, created_edges = self.linker.materialize(
                entities,
                relations,
                decision.selected_profile,
                self.state.nodes,
            )
            for edge in created_edges:
                self.state.edges[edge.edge_id] = edge
        return self.state

    def analyze_quality(self, questions: list[Question]) -> GraphState:
        self.state.quality_issues = self.quality_analyzer.analyze_global(self.state.nodes, self.state.edges)
        self.state.query_support = self.quality_analyzer.analyze_query_support(questions, self.state.edges)

        for issue in self.state.quality_issues:
            if issue.object_type == "node" and issue.object_id in self.state.nodes:
                self.state.nodes[issue.object_id].metadata.setdefault("quality_issues", []).append(issue.issue_type)
            if issue.object_type == "edge" and issue.object_id in self.state.edges:
                self.state.edges[issue.object_id].metadata.setdefault("quality_issues", []).append(issue.issue_type)

        for record in self.state.query_support.values():
            for edge_id in record.supporting_edge_ids:
                self.state.edges[edge_id].metadata.setdefault("query_support", []).append(record.question_id)

        return self.state

    def answer_question(self, question: Question) -> AnswerResult:
        query_support = self.state.query_support.get(question.question_id)
        selected_edges = self.selector.select_edges(question, self.state.edges)

        quality_notes: list[str] = []
        if query_support and not query_support.answerable:
            quality_notes.append(
                f"Missing query path requirements: {', '.join(query_support.missing_requirements)}"
            )
        if not selected_edges:
            quality_notes.append("No high-quality supporting edges were selected.")

        answer, confidence = self.answer_generator.generate(question, selected_edges, self.state.nodes)

        if query_support:
            confidence *= max(0.25, query_support.support_score)

        return AnswerResult(
            question_id=question.question_id,
            answer=answer,
            confidence=round(confidence, 3),
            profile_used=question.target_profile,
            selected_edge_ids=[edge.edge_id for edge in selected_edges],
            quality_notes=quality_notes,
        )

    def answer_questions(self, questions: list[Question]) -> list[AnswerResult]:
        return [self.answer_question(question) for question in questions]
