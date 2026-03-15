from __future__ import annotations

from collections import Counter, defaultdict

from .fibo_profiles import FIBO_PROFILES
from .models import GraphEdge, GraphNode, QualityIssue, QuerySupportRecord, Question


class GraphQualityAnalyzer:
    template_requirements = {
        "who-serves-on-committee": ("serves_on_committee",),
        "metric-for-period": ("reports_metric", "reported_for_period"),
        "shareholder-return-policy": ("announced_repurchase", "declared_dividend"),
        "governance-query": ("holds_role", "serves_on_committee"),
        "financials-query": ("reports_metric", "reported_for_period"),
        "shareholder-return-query": ("announced_repurchase", "declared_dividend"),
    }

    def analyze_global(
        self,
        nodes: dict[str, GraphNode],
        edges: dict[str, GraphEdge],
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []

        alias_counter = Counter()
        for node in nodes.values():
            for alias in node.aliases:
                alias_counter[alias.lower()] += 1

        for alias, count in alias_counter.items():
            if count > 1:
                issues.append(
                    QualityIssue(
                        issue_type="duplicate_alias",
                        severity="medium",
                        message=f"Alias '{alias}' appears in multiple canonical nodes.",
                        object_type="node_alias",
                        object_id=alias,
                        metadata={"count": count},
                    )
                )

        adjacency = defaultdict(set)
        for edge in edges.values():
            missing_endpoints = [
                node_id
                for node_id in (edge.source_node_id, edge.target_node_id)
                if node_id not in nodes
            ]
            if missing_endpoints:
                issues.append(
                    QualityIssue(
                        issue_type="dangling_edge",
                        severity="high",
                        message="Edge references node ids that are missing from the graph state.",
                        object_type="edge",
                        object_id=edge.edge_id,
                        metadata={"missing_node_ids": missing_endpoints},
                    )
                )
                continue

            adjacency[edge.source_node_id].add(edge.target_node_id)
            adjacency[edge.target_node_id].add(edge.source_node_id)

            allowed = FIBO_PROFILES[edge.profile].relation_types
            if edge.relation_type not in allowed:
                issues.append(
                    QualityIssue(
                        issue_type="schema_violation",
                        severity="high",
                        message=f"Edge relation '{edge.relation_type}' is not allowed in profile '{edge.profile}'.",
                        object_type="edge",
                        object_id=edge.edge_id,
                    )
                )

        for node_id in nodes:
            if len(adjacency[node_id]) == 0:
                issues.append(
                    QualityIssue(
                        issue_type="disconnected_node",
                        severity="medium",
                        message="Node is disconnected from the graph.",
                        object_type="node",
                        object_id=node_id,
                    )
                )

        return issues

    def analyze_query_support(
        self,
        questions: list[Question],
        edges: dict[str, GraphEdge],
        nodes: dict[str, GraphNode] | None = None,
    ) -> dict[str, QuerySupportRecord]:
        support_records: dict[str, QuerySupportRecord] = {}

        for question in questions:
            required_relations = tuple(question.required_relations) or self.template_requirements.get(question.query_template, ())
            supporting = [
                edge
                for edge in edges.values()
                if edge.profile == question.target_profile and edge.relation_type in required_relations
            ]
            seen_relations = {edge.relation_type for edge in supporting if edge.relation_type in required_relations}
            matched_entity_types = self._matched_entity_types(supporting, nodes) if nodes else set()
            filled_slots = self._filled_slots(question, supporting, nodes) if nodes else set()

            requirements: list[str] = []
            requirements.extend(f"relation:{relation}" for relation in required_relations)
            requirements.extend(f"entity_type:{entity_type}" for entity_type in question.required_entity_types)
            requirements.extend(f"slot:{slot}" for slot in question.focus_slots)

            matched_requirements = {
                f"relation:{relation}" for relation in seen_relations
            }
            matched_requirements.update(
                f"entity_type:{entity_type}"
                for entity_type in matched_entity_types
                if entity_type in question.required_entity_types
            )
            matched_requirements.update(
                f"slot:{slot}"
                for slot in filled_slots
                if slot in question.focus_slots
            )

            missing = [requirement for requirement in requirements if requirement not in matched_requirements]
            support_score = 0.0 if not requirements else len(matched_requirements) / len(requirements)
            support_records[question.question_id] = QuerySupportRecord(
                question_id=question.question_id,
                query_template=question.query_template,
                target_profile=question.target_profile,
                intent_id=question.intent_id or question.query_template,
                support_score=support_score,
                answerable=len(missing) == 0,
                supporting_edge_ids=[edge.edge_id for edge in supporting],
                missing_requirements=missing,
                matched_relations=sorted(seen_relations),
                matched_entity_types=sorted(matched_entity_types),
                filled_slots=sorted(filled_slots),
            )

        return support_records

    def _matched_entity_types(
        self,
        supporting: list[GraphEdge],
        nodes: dict[str, GraphNode] | None,
    ) -> set[str]:
        if not nodes:
            return set()
        matched: set[str] = set()
        for edge in supporting:
            source = nodes.get(edge.source_node_id)
            target = nodes.get(edge.target_node_id)
            if source is not None:
                matched.add(source.entity_type)
            if target is not None:
                matched.add(target.entity_type)
        return matched

    def _filled_slots(
        self,
        question: Question,
        supporting: list[GraphEdge],
        nodes: dict[str, GraphNode] | None,
    ) -> set[str]:
        if not nodes:
            return set()

        if question.intent_id == "governance_role_transition":
            return self._slots_for_governance_role_transition(supporting, nodes)
        if question.intent_id == "governance_committee_oversight":
            return self._slots_for_governance_committee_oversight(supporting, nodes)
        if question.intent_id == "governance_board_structure":
            return self._slots_for_governance_board_structure(supporting, nodes)
        if question.intent_id == "governance_executive_roles":
            return self._slots_for_governance_executive_roles(supporting, nodes)
        if question.intent_id == "financials_liquidity_breakdown":
            return self._slots_for_financial_liquidity(supporting, nodes)
        if question.intent_id == "financials_metric_delta":
            return self._slots_for_financial_metric_delta(supporting, nodes)
        if question.intent_id == "financials_profitability_trend":
            return self._slots_for_financial_profitability(supporting, nodes)
        if question.intent_id == "financials_debt_profile":
            return self._slots_for_financial_debt(supporting, nodes)
        if question.intent_id == "shareholder_return_authorization":
            return self._slots_for_shareholder_authorization(supporting, nodes)
        if question.intent_id == "shareholder_return_activity":
            return self._slots_for_shareholder_activity(supporting, nodes)
        if question.intent_id == "shareholder_return_policy_mix":
            return self._slots_for_shareholder_policy_mix(supporting, nodes)
        if question.intent_id == "shareholder_return_tax_impact":
            return self._slots_for_shareholder_tax_impact(supporting, nodes)
        return self._slots_by_hint_matching(question, supporting, nodes)

    def _slots_by_hint_matching(
        self,
        question: Question,
        supporting: list[GraphEdge],
        nodes: dict[str, GraphNode],
    ) -> set[str]:
        phrases = []
        for edge in supporting:
            source = nodes.get(edge.source_node_id)
            target = nodes.get(edge.target_node_id)
            if source is not None:
                phrases.append(source.name.lower())
            if target is not None:
                phrases.append(target.name.lower())
            phrases.append(edge.relation_type.lower())

        joined = " ".join(phrases)
        filled: set[str] = set()
        for slot in question.focus_slots:
            hints = tuple(question.focus_slot_hints.get(slot, ()))
            if not hints:
                continue
            if any(hint in joined for hint in hints):
                filled.add(slot)
        return filled

    def _slots_for_governance_role_transition(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        filled: set[str] = set()
        role_counts: Counter[str] = Counter()
        for edge in supporting:
            if edge.relation_type != "holds_role":
                continue
            source = nodes.get(edge.source_node_id)
            target = nodes.get(edge.target_node_id)
            if source is not None and source.entity_type == "Person":
                filled.add("person")
                role_counts[source.node_id] += 1
            if target is not None and target.entity_type == "OfficerRole":
                filled.add("current_role")
        if any(count >= 2 for count in role_counts.values()):
            filled.add("prior_role")
        return filled

    def _slots_for_governance_committee_oversight(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        filled = self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="governance",
                ground_truth_answer="",
                focus_slots=("committee", "oversight_roles", "oversight_topic"),
                focus_slot_hints={
                    "committee": ("committee", "board"),
                    "oversight_roles": ("cio", "ciso", "chief", "officer"),
                    "oversight_topic": ("cyber", "risk", "security"),
                },
            ),
            supporting,
            nodes,
        )
        return filled

    def _slots_for_governance_board_structure(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        filled = self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="governance",
                ground_truth_answer="",
                focus_slots=("board_leadership", "directors"),
                focus_slot_hints={
                    "board_leadership": ("chair", "lead independent director", "board"),
                    "directors": ("director", "board"),
                },
            ),
            supporting,
            nodes,
        )
        return filled

    def _slots_for_governance_executive_roles(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        filled = self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="governance",
                ground_truth_answer="",
                focus_slots=("executives", "roles"),
                focus_slot_hints={
                    "executives": ("chief", "executive", "president", "officer"),
                    "roles": ("chief", "head", "president", "director", "counsel"),
                },
            ),
            supporting,
            nodes,
        )
        return filled

    def _slots_for_financial_liquidity(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="financials",
                ground_truth_answer="",
                focus_slots=("cash", "investments", "credit_facility", "period"),
                focus_slot_hints={
                    "cash": ("cash", "cash equivalents"),
                    "investments": ("investment", "treasury"),
                    "credit_facility": ("facility", "credit", "revolving"),
                    "period": ("202", "q1", "q2", "q3", "q4", "january", "july", "october", "december"),
                },
            ),
            supporting,
            nodes,
        )

    def _slots_for_financial_metric_delta(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        filled = self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="financials",
                ground_truth_answer="",
                focus_slots=("metric", "current_period", "prior_period", "value_delta"),
                focus_slot_hints={
                    "metric": ("revenue", "income", "cash", "eps", "profit", "margin"),
                    "current_period": ("2024", "2023", "2022", "2021", "q1", "q2", "q3", "q4"),
                    "prior_period": ("2024", "2023", "2022", "2021", "q1", "q2", "q3", "q4"),
                    "value_delta": ("increase", "decrease", "delta"),
                },
            ),
            supporting,
            nodes,
        )
        reporting_period_nodes = {
            edge.target_node_id
            for edge in supporting
            if edge.relation_type == "reported_for_period"
        }
        if len(reporting_period_nodes) >= 2:
            filled.add("current_period")
            filled.add("prior_period")
        return filled

    def _slots_for_financial_profitability(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="financials",
                ground_truth_answer="",
                focus_slots=("metric", "periods", "values"),
                focus_slot_hints={
                    "metric": ("profit", "margin", "income", "eps", "earnings"),
                    "periods": ("2024", "2023", "2022", "2021"),
                    "values": (),
                },
            ),
            supporting,
            nodes,
        )

    def _slots_for_financial_debt(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="financials",
                ground_truth_answer="",
                focus_slots=("debt_instrument", "period", "amount"),
                focus_slot_hints={
                    "debt_instrument": ("note", "debt", "loan"),
                    "period": ("2024", "2023", "2022", "2021"),
                    "amount": (),
                },
            ),
            supporting,
            nodes,
        )

    def _slots_for_shareholder_authorization(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="shareholder_return",
                ground_truth_answer="",
                focus_slots=("program", "authorized_amount", "remaining_capacity"),
                focus_slot_hints={
                    "program": ("program", "authorization"),
                    "authorized_amount": (),
                    "remaining_capacity": ("remaining", "available", "capacity"),
                },
            ),
            supporting,
            nodes,
        )

    def _slots_for_shareholder_activity(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="shareholder_return",
                ground_truth_answer="",
                focus_slots=("program", "repurchase_amount", "price_or_volume"),
                focus_slot_hints={
                    "program": ("repurchase", "buyback", "program"),
                    "repurchase_amount": (),
                    "price_or_volume": ("share", "shares", "price"),
                },
            ),
            supporting,
            nodes,
        )

    def _slots_for_shareholder_policy_mix(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="shareholder_return",
                ground_truth_answer="",
                focus_slots=("repurchase_program", "dividend_program"),
                focus_slot_hints={
                    "repurchase_program": ("repurchase", "buyback", "program"),
                    "dividend_program": ("dividend",),
                },
            ),
            supporting,
            nodes,
        )

    def _slots_for_shareholder_tax_impact(self, supporting: list[GraphEdge], nodes: dict[str, GraphNode]) -> set[str]:
        return self._slots_by_hint_matching(
            Question(
                question_id="",
                question="",
                query_template="",
                target_profile="shareholder_return",
                ground_truth_answer="",
                focus_slots=("tax_factor", "repurchase_program", "dividend_program"),
                focus_slot_hints={
                    "tax_factor": ("tax", "excise"),
                    "repurchase_program": ("repurchase", "buyback", "program"),
                    "dividend_program": ("dividend",),
                },
            ),
            supporting,
            nodes,
        )
