from __future__ import annotations

from typing import Dict, List, Union

from openai import OpenAI

from framework.llm_client import structured_completion
from framework.llm_models import OntologySelectionResponse
from framework.schema import OntologySelection


DEFAULT_FIBO_PACKS: Dict[str, Dict[str, Union[List[str], str]]] = {
    "governance": {
        "modules": [
            "fibo-be-le-lp",
            "fibo-fbc-fct-ra",
        ],
        "classes": [
            "LegalEntity",
            "CorporateBody",
            "BoardOfDirectors",
            "RegulatoryAgency",
        ],
        "reason": "Governance questions tend to center on legal entities, roles, and oversight structures.",
    },
    "financials": {
        "modules": [
            "fibo-fbc-fi-fi",
            "fibo-fnd-acc-cur",
            "fibo-fnd-arr-rep",
        ],
        "classes": [
            "FinancialInstrument",
            "MonetaryAmount",
            "Revenue",
            "Expense",
        ],
        "reason": "Financials questions often involve accounting amounts, instrument context, and reporting concepts.",
    },
    "shareholder return": {
        "modules": [
            "fibo-sec-eq-eq",
            "fibo-fbc-fi-ip",
        ],
        "classes": [
            "EquityInstrument",
            "Shareholder",
            "Dividend",
            "StockRepurchase",
        ],
        "reason": "Shareholder return questions commonly reference equity instruments, holders, dividends, and buybacks.",
    },
    "accounting": {
        "modules": [
            "fibo-fnd-acc-aeq",
            "fibo-fbc-pas-fpas",
        ],
        "classes": [
            "Revenue",
            "Expense",
            "MonetaryAmount",
            "AccountingPolicy",
            "ContractualObligation",
        ],
        "reason": "Accounting questions involve revenue recognition, expenses, monetary amounts, and accounting policies.",
    },
    "legal": {
        "modules": [
            "fibo-be-le-lp",
            "fibo-be-ge-ge",
            "fibo-fbc-fct-cajr",
        ],
        "classes": [
            "LegalEntity",
            "CorporateBody",
            "LegalProceeding",
            "Court",
            "GovernmentEntity",
            "Legislation",
        ],
        "reason": "Legal questions reference corporations, courts, legal proceedings, government entities, and statutes.",
    },
    "risk": {
        "modules": [
            "fibo-fnd-org-fm",
            "fibo-fbc-fct-rga",
        ],
        "classes": [
            "LegalEntity",
            "CorporateBody",
            "BoardOfDirectors",
            "CorporateOfficer",
            "AuditCommittee",
            "ComplianceStandard",
        ],
        "reason": "Risk questions reference corporate governance bodies, officers, compliance standards, and oversight roles.",
    },
    "footnotes": {
        "modules": [
            "fibo-fnd-acc-cur",
            "fibo-fnd-agr-agr",
            "fibo-fbc-fi-fi",
        ],
        "classes": [
            "MonetaryAmount",
            "Expense",
            "CapitalExpenditure",
            "ContractualObligation",
            "FinancialInstrument",
        ],
        "reason": "Footnotes questions involve capital expenditures, monetary amounts, contractual agreements, and financial instruments.",
    },
    "company overview": {
        "modules": [
            "fibo-fbc-fct-breg",
            "fibo-fnd-org-fm",
        ],
        "classes": [
            "LegalEntity",
            "CorporateBody",
            "BusinessSegment",
            "GeographicRegion",
        ],
        "reason": "Company overview questions reference named corporations, business segments, and geographic operations.",
    },
}


def _empty_selection(*, prompt_template_id: str, mode: str, notes: list[str] | None = None) -> OntologySelection:
    return OntologySelection(
        selection_mode=mode,
        selected_pack_keys=[],
        selected_modules=[],
        selected_classes=[],
        rejected_pack_keys=[],
        selection_reason="Ontology guidance disabled for this run.",
        selection_confidence=1.0 if mode == "none" else 0.0,
        prompt_template_id=prompt_template_id,
        notes=notes or [],
    )


def _static_pack_selection(*, category: str, prompt_template_id: str, mode: str = "static") -> OntologySelection:
    pack = DEFAULT_FIBO_PACKS.get(category.strip().lower())
    if pack is None:
        return OntologySelection(
            selection_mode=mode,
            selected_pack_keys=[],
            selected_modules=[],
            selected_classes=[],
            rejected_pack_keys=[],
            selection_reason="No category-specific FIBO pack matched; use a minimal generic finance pack.",
            selection_confidence=0.25,
            prompt_template_id=prompt_template_id,
            notes=["Category did not match the current static packs. Review dataset categories after smoke run."],
        )

    return OntologySelection(
        selection_mode=mode,
        selected_pack_keys=[category.strip().lower()],
        selected_modules=list(pack["modules"]),
        selected_classes=list(pack["classes"]),
        rejected_pack_keys=[],
        selection_reason=str(pack["reason"]),
        selection_confidence=0.7,
        prompt_template_id=prompt_template_id,
        notes=["Static pack selection enabled for observability-first smoke runs."],
    )


def _merge_pack_selection(
    *,
    prompt_template_id: str,
    mode: str,
    selected_pack_keys: list[str],
    rejected_pack_keys: list[str],
    selection_reason: str,
    selection_confidence: float | None,
    selection_prompt_preview: str = "",
    notes: list[str] | None = None,
) -> OntologySelection:
    modules: list[str] = []
    classes: list[str] = []
    seen_modules: set[str] = set()
    seen_classes: set[str] = set()

    for pack_key in selected_pack_keys:
        pack = DEFAULT_FIBO_PACKS.get(pack_key)
        if pack is None:
            continue
        for module_name in pack["modules"]:
            module_name = str(module_name)
            if module_name not in seen_modules:
                modules.append(module_name)
                seen_modules.add(module_name)
        for class_name in pack["classes"]:
            class_name = str(class_name)
            if class_name not in seen_classes:
                classes.append(class_name)
                seen_classes.add(class_name)

    return OntologySelection(
        selection_mode=mode,
        selected_pack_keys=selected_pack_keys,
        selected_modules=modules,
        selected_classes=classes,
        rejected_pack_keys=rejected_pack_keys,
        selection_reason=selection_reason,
        selection_confidence=selection_confidence,
        prompt_template_id=prompt_template_id,
        selection_prompt_preview=selection_prompt_preview,
        notes=notes or [],
    )


def build_ontology_selection_prompt(*, max_packs: int = 3) -> str:
    lines = [
        "You are an ontology slice selector for a financial entity extraction and linking pipeline.",
        "",
        "Task:",
        "- Read the question and evidence text.",
        "- Select the smallest ontology pack subset that gives strong semantic coverage for downstream extraction and linking.",
        f"- Select between 0 and {max_packs} pack keys from the provided candidate list.",
        "- Prefer precise, task-relevant packs over broad or noisy coverage.",
        "- If the evidence does not clearly fit any pack, return an empty selection with an explanatory reason.",
        "",
        "Important guidance:",
        "- Ontology selection is part of the indexing pipeline, not external preprocessing.",
        "- Use the question to understand the task intent and the evidence text to confirm the domain context.",
        "- Avoid selecting packs just because they are vaguely finance-related.",
        "",
        "Candidate packs:",
    ]
    for pack_key, pack in DEFAULT_FIBO_PACKS.items():
        modules = ", ".join(str(item) for item in pack["modules"])
        classes = ", ".join(str(item) for item in pack["classes"])
        lines.extend(
            [
                f"- {pack_key}",
                f"  Modules: {modules}",
                f"  Classes: {classes}",
                f"  When useful: {pack['reason']}",
            ]
        )

    lines.extend(
        [
            "",
            "Return JSON with selected_pack_keys, rejected_pack_keys, selection_reason, selection_confidence, and notes.",
        ]
    )
    return "\n".join(lines)


def _dynamic_pack_selection(
    *,
    prompt_template_id: str,
    question: str,
    references_text: str,
    openai_client: OpenAI | None,
    model: str,
    max_packs: int = 3,
    fallback_category: str = "",
) -> OntologySelection:
    selection_prompt = build_ontology_selection_prompt(max_packs=max_packs)
    if openai_client is None or not model:
        fallback = _static_pack_selection(
            category=fallback_category,
            prompt_template_id=prompt_template_id,
            mode="dynamic",
        )
        fallback.selection_prompt_preview = selection_prompt
        fallback.notes.append("Dynamic selection requested but OpenAI client/model was unavailable; used static fallback.")
        return fallback

    user_prompt = (
        f"Question:\n{question or '(empty)'}\n\n"
        f"Evidence text:\n{references_text[:5000] or '(empty)'}\n"
    )

    try:
        result = structured_completion(
            openai_client,
            model=model,
            system_prompt=selection_prompt,
            user_prompt=user_prompt,
            response_model=OntologySelectionResponse,
        )
        response = result.parsed
    except Exception as exc:
        fallback = _static_pack_selection(
            category=fallback_category,
            prompt_template_id=prompt_template_id,
            mode="dynamic",
        )
        fallback.selection_prompt_preview = selection_prompt
        fallback.notes.append(f"Dynamic selection failed; used static fallback: {exc}")
        return fallback

    selected_pack_keys: list[str] = []
    rejected_pack_keys: list[str] = []
    for pack_key in response.selected_pack_keys:
        normalized = pack_key.strip().lower()
        if normalized in DEFAULT_FIBO_PACKS and normalized not in selected_pack_keys:
            selected_pack_keys.append(normalized)
    for pack_key in response.rejected_pack_keys:
        normalized = pack_key.strip().lower()
        if normalized in DEFAULT_FIBO_PACKS and normalized not in rejected_pack_keys:
            rejected_pack_keys.append(normalized)

    if len(selected_pack_keys) > max_packs:
        selected_pack_keys = selected_pack_keys[:max_packs]

    if not selected_pack_keys:
        fallback = _static_pack_selection(
            category=fallback_category,
            prompt_template_id=prompt_template_id,
            mode="dynamic",
        )
        fallback.selection_prompt_preview = selection_prompt
        fallback.notes.extend(response.notes)
        fallback.notes.append("Dynamic selection returned no valid packs; used static fallback.")
        return fallback

    return _merge_pack_selection(
        prompt_template_id=prompt_template_id,
        mode="dynamic",
        selected_pack_keys=selected_pack_keys,
        rejected_pack_keys=rejected_pack_keys,
        selection_reason=response.selection_reason,
        selection_confidence=response.selection_confidence,
        selection_prompt_preview=selection_prompt,
        notes=response.notes,
    )


def _all_packs_selection(*, prompt_template_id: str) -> OntologySelection:
    """Select ALL packs with full metadata (Condition D: rule off, meta on)."""
    all_keys = list(DEFAULT_FIBO_PACKS.keys())
    return _merge_pack_selection(
        prompt_template_id=prompt_template_id,
        mode="all_packs",
        selected_pack_keys=all_keys,
        rejected_pack_keys=[],
        selection_reason="All packs selected (ablation Condition D: no category filtering, full metadata).",
        selection_confidence=1.0,
        notes=["Ablation Condition D: all packs loaded without category filtering."],
    )


def select_ontology_context(
    *,
    category: str,
    prompt_template_id: str,
    mode: str = "all_packs",
    question: str = "",
    references_text: str = "",
    openai_client: OpenAI | None = None,
    model: str = "",
    max_packs: int = 3,
) -> OntologySelection:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "none":
        return _empty_selection(
            prompt_template_id=prompt_template_id,
            mode="none",
            notes=["No ontology guidance selected; this run measures model-only extraction/linking."],
        )
    if normalized_mode == "dynamic":
        return _dynamic_pack_selection(
            prompt_template_id=prompt_template_id,
            question=question,
            references_text=references_text,
            openai_client=openai_client,
            model=model,
            max_packs=max_packs,
            fallback_category=category,
        )
    if normalized_mode == "all_packs":
        return _all_packs_selection(prompt_template_id=prompt_template_id)
    if normalized_mode == "rule_only":
        # Same as static selection, but prompt_rules will strip metadata
        return _static_pack_selection(
            category=category,
            prompt_template_id=prompt_template_id,
            mode="rule_only",
        )
    return _static_pack_selection(
        category=category,
        prompt_template_id=prompt_template_id,
        mode="static",
    )
