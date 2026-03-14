from __future__ import annotations

from collections import Counter, defaultdict

from .fibo_profiles import FIBO_PROFILES
from .models import GraphEdge, GraphNode, QualityIssue, QuerySupportRecord, Question


class GraphQualityAnalyzer:
    template_requirements = {
        "who-serves-on-committee": ("serves_on_committee",),
        "metric-for-period": ("reports_metric", "reported_for_period"),
        "shareholder-return-policy": ("announced_repurchase", "declared_dividend"),
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
    ) -> dict[str, QuerySupportRecord]:
        support_records: dict[str, QuerySupportRecord] = {}

        for question in questions:
            required = self.template_requirements.get(question.query_template, ())
            supporting = [
                edge
                for edge in edges.values()
                if edge.profile == question.target_profile and edge.relation_type in required
            ]
            seen_relations = {edge.relation_type for edge in supporting}
            missing = [relation for relation in required if relation not in seen_relations]
            support_score = 0.0 if not required else len(seen_relations) / len(required)
            support_records[question.question_id] = QuerySupportRecord(
                question_id=question.question_id,
                query_template=question.query_template,
                target_profile=question.target_profile,
                support_score=support_score,
                answerable=len(missing) == 0,
                supporting_edge_ids=[edge.edge_id for edge in supporting],
                missing_requirements=missing,
            )

        return support_records
