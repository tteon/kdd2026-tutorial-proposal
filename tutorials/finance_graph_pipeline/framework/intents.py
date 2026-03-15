from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    profile: str
    required_relations: tuple[str, ...]
    required_entity_types: tuple[str, ...]
    focus_slots: tuple[str, ...]
    trigger_keywords: tuple[str, ...] = field(default_factory=tuple)
    focus_slot_hints: dict[str, tuple[str, ...]] = field(default_factory=dict)


INTENT_CATALOG: dict[str, tuple[IntentSpec, ...]] = {
    "governance": (
        IntentSpec(
            intent_id="governance_role_transition",
            profile="governance",
            required_relations=("holds_role",),
            required_entity_types=("Person", "OfficerRole"),
            focus_slots=("person", "current_role", "prior_role"),
            trigger_keywords=("transition", "career", "progression", "tenure", "previously", "role"),
            focus_slot_hints={
                "person": (),
                "current_role": ("chief", "head", "officer", "president", "director", "chair"),
                "prior_role": ("previous", "prior", "formerly", "access solutions", "information solutions"),
            },
        ),
        IntentSpec(
            intent_id="governance_committee_oversight",
            profile="governance",
            required_relations=("serves_on_committee", "holds_role"),
            required_entity_types=("Committee", "Person"),
            focus_slots=("committee", "oversight_roles", "oversight_topic"),
            trigger_keywords=("committee", "audit", "oversight", "cybersecurity", "risk", "cio", "ciso"),
            focus_slot_hints={
                "committee": ("committee", "board"),
                "oversight_roles": ("cio", "ciso", "chief", "officer"),
                "oversight_topic": ("cyber", "risk", "security"),
            },
        ),
        IntentSpec(
            intent_id="governance_board_structure",
            profile="governance",
            required_relations=("holds_role", "serves_on_committee"),
            required_entity_types=("Person", "Board"),
            focus_slots=("board_leadership", "directors"),
            trigger_keywords=("board", "director", "independent", "chair", "separation", "composition"),
            focus_slot_hints={
                "board_leadership": ("chair", "lead independent director", "board"),
                "directors": ("director", "board"),
            },
        ),
        IntentSpec(
            intent_id="governance_executive_roles",
            profile="governance",
            required_relations=("holds_role",),
            required_entity_types=("Person", "OfficerRole"),
            focus_slots=("executives", "roles"),
            trigger_keywords=("cfo", "coo", "general counsel", "legal", "executive", "officer", "ops"),
            focus_slot_hints={
                "executives": ("chief", "executive", "president", "officer"),
                "roles": ("chief", "head", "president", "director", "counsel"),
            },
        ),
    ),
    "financials": (
        IntentSpec(
            intent_id="financials_liquidity_breakdown",
            profile="financials",
            required_relations=("reports_metric", "reported_for_period"),
            required_entity_types=("FinancialMetric", "ReportingPeriod", "MonetaryAmount"),
            focus_slots=("cash", "investments", "credit_facility", "period"),
            trigger_keywords=("liquidity", "cash", "cash equivalents", "short-term", "short term", "coverage", "credit", "facility"),
            focus_slot_hints={
                "cash": ("cash", "cash equivalents"),
                "investments": ("investment", "treasury"),
                "credit_facility": ("facility", "credit", "revolving"),
                "period": (),
            },
        ),
        IntentSpec(
            intent_id="financials_metric_delta",
            profile="financials",
            required_relations=("reports_metric", "reported_for_period"),
            required_entity_types=("FinancialMetric", "ReportingPeriod", "MonetaryAmount"),
            focus_slots=("metric", "current_period", "prior_period", "value_delta"),
            trigger_keywords=("delta", "change", "growth", "trend", "increase", "decrease", "from", "to"),
            focus_slot_hints={
                "metric": ("revenue", "income", "cash", "eps", "profit", "margin"),
                "current_period": (),
                "prior_period": (),
                "value_delta": ("increase", "decrease", "delta"),
            },
        ),
        IntentSpec(
            intent_id="financials_profitability_trend",
            profile="financials",
            required_relations=("reports_metric", "reported_for_period"),
            required_entity_types=("FinancialMetric", "ReportingPeriod", "MonetaryAmount"),
            focus_slots=("metric", "periods", "values"),
            trigger_keywords=("profitability", "margin", "gross profit", "operating income", "earnings", "eps", "net income"),
            focus_slot_hints={
                "metric": ("profit", "margin", "income", "eps", "earnings"),
                "periods": (),
                "values": (),
            },
        ),
        IntentSpec(
            intent_id="financials_debt_profile",
            profile="financials",
            required_relations=("reports_metric", "reported_for_period"),
            required_entity_types=("FinancialMetric", "ReportingPeriod", "MonetaryAmount"),
            focus_slots=("debt_instrument", "period", "amount"),
            trigger_keywords=("debt", "notes", "interest", "repaid", "repay", "long-term debt", "lt debt"),
            focus_slot_hints={
                "debt_instrument": ("notes", "debt", "term loan"),
                "period": (),
                "amount": (),
            },
        ),
    ),
    "shareholder_return": (
        IntentSpec(
            intent_id="shareholder_return_authorization",
            profile="shareholder_return",
            required_relations=("announced_repurchase",),
            required_entity_types=("RepurchaseProgram", "MonetaryAmount"),
            focus_slots=("program", "authorized_amount", "remaining_capacity"),
            trigger_keywords=("authorization", "authorized", "remaining", "balance", "capacity", "available"),
            focus_slot_hints={
                "program": ("program", "authorization"),
                "authorized_amount": (),
                "remaining_capacity": ("remaining", "available", "capacity"),
            },
        ),
        IntentSpec(
            intent_id="shareholder_return_activity",
            profile="shareholder_return",
            required_relations=("announced_repurchase",),
            required_entity_types=("RepurchaseProgram", "MonetaryAmount"),
            focus_slots=("program", "repurchase_amount", "price_or_volume"),
            trigger_keywords=("repurchase", "buyback", "shares", "average price", "cost", "timing", "purchased"),
            focus_slot_hints={
                "program": ("repurchase", "buyback", "program"),
                "repurchase_amount": (),
                "price_or_volume": ("average price", "share", "shares"),
            },
        ),
        IntentSpec(
            intent_id="shareholder_return_policy_mix",
            profile="shareholder_return",
            required_relations=("announced_repurchase", "declared_dividend"),
            required_entity_types=("RepurchaseProgram", "Dividend"),
            focus_slots=("repurchase_program", "dividend_program"),
            trigger_keywords=("dividend", "policy", "capital return", "shareholder return", "interplay", "drivers"),
            focus_slot_hints={
                "repurchase_program": ("repurchase", "buyback", "program"),
                "dividend_program": ("dividend",),
            },
        ),
        IntentSpec(
            intent_id="shareholder_return_tax_impact",
            profile="shareholder_return",
            required_relations=("announced_repurchase", "declared_dividend"),
            required_entity_types=("RepurchaseProgram", "Dividend"),
            focus_slots=("tax_factor", "repurchase_program", "dividend_program"),
            trigger_keywords=("excise tax", "inflation reduction act", "ira", "tax impact"),
            focus_slot_hints={
                "tax_factor": ("tax", "excise"),
                "repurchase_program": ("repurchase", "buyback", "program"),
                "dividend_program": ("dividend",),
            },
        ),
    ),
}


DEFAULT_INTENTS = {
    "governance": "governance_board_structure",
    "financials": "financials_metric_delta",
    "shareholder_return": "shareholder_return_activity",
}


def get_intent_spec(profile: str, intent_id: str) -> IntentSpec:
    for spec in INTENT_CATALOG.get(profile, ()):
        if spec.intent_id == intent_id:
            return spec
    raise KeyError(f"Unknown intent '{intent_id}' for profile '{profile}'")


def infer_intent(profile: str, question_text: str) -> IntentSpec:
    normalized = question_text.lower()
    best_spec: IntentSpec | None = None
    best_score = -1
    for spec in INTENT_CATALOG.get(profile, ()):
        score = sum(1 for keyword in spec.trigger_keywords if keyword in normalized)
        if score > best_score:
            best_spec = spec
            best_score = score
    if best_spec is not None and best_score > 0:
        return best_spec
    return get_intent_spec(profile, DEFAULT_INTENTS[profile])
