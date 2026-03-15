from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FiboProfile:
    name: str
    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    required_query_templates: tuple[str, ...]


FIBO_PROFILES: dict[str, FiboProfile] = {
    "governance": FiboProfile(
        name="governance",
        entity_types=("LegalEntity", "Person", "OfficerRole", "Board", "Committee"),
        relation_types=("holds_role", "chairs", "serves_on_committee"),
        required_query_templates=("who-serves-on-committee", "governance-query"),
    ),
    "financials": FiboProfile(
        name="financials",
        entity_types=("LegalEntity", "FinancialMetric", "ReportingPeriod", "BusinessSegment", "MonetaryAmount"),
        relation_types=("reports_metric", "reported_for_period", "segment_contribution"),
        required_query_templates=("metric-for-period", "financials-query"),
    ),
    "shareholder_return": FiboProfile(
        name="shareholder_return",
        entity_types=("LegalEntity", "RepurchaseProgram", "Dividend", "ShareClass", "MonetaryAmount"),
        relation_types=("announced_repurchase", "declared_dividend", "applies_to_share_class"),
        required_query_templates=("shareholder-return-policy", "shareholder-return-query"),
    ),
}


SECTION_TO_PROFILE = {
    "governance": "governance",
    "financials": "financials",
    "shareholder return": "shareholder_return",
}
