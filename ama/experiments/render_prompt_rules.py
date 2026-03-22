from __future__ import annotations

import argparse
import json

from framework.llm_client import get_openai_client
from framework.ontology import select_ontology_context
from framework.prompt_rules import build_prompt_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ontology-selected prompt rules for inspection."
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--ontology-mode", choices=["none", "static", "dynamic", "rule_only", "all_packs"], default="all_packs")
    parser.add_argument("--prompt-template-id", default="finder_fibo_v1")
    parser.add_argument("--question", default="")
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    openai_client = get_openai_client() if args.ontology_mode == "dynamic" else None
    selection = select_ontology_context(
        category=args.category,
        prompt_template_id=args.prompt_template_id,
        mode=args.ontology_mode,
        question=args.question,
        references_text=args.reference_text,
        openai_client=openai_client,
        model=args.model,
    )
    context = build_prompt_context(selection)

    if args.json:
        payload = {
            "ontology_selection": {
                "selection_mode": selection.selection_mode,
                "selected_pack_keys": selection.selected_pack_keys,
                "selected_modules": selection.selected_modules,
                "selected_classes": selection.selected_classes,
                "rejected_pack_keys": selection.rejected_pack_keys,
                "selection_reason": selection.selection_reason,
                "selection_confidence": selection.selection_confidence,
                "selection_prompt_preview": selection.selection_prompt_preview,
            },
            "rules": [
                {
                    "concept_id": rule.concept_id,
                    "label": rule.label,
                    "definition": rule.definition,
                    "synonyms": rule.synonyms,
                    "semantic_features": rule.semantic_features,
                    "soft_signals": rule.soft_signals,
                    "extraction_hint": rule.extraction_hint,
                    "linking_hint": rule.linking_hint,
                }
                for rule in context.rules
            ],
            "notes": context.notes,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("=== Ontology Selection ===")
    print(f"mode: {selection.selection_mode}")
    print(f"packs: {', '.join(selection.selected_pack_keys) or '(none)'}")
    print(f"modules: {', '.join(selection.selected_modules) or '(none)'}")
    print(f"classes: {', '.join(selection.selected_classes) or '(none)'}")
    print(f"reason: {selection.selection_reason}")
    if selection.selection_prompt_preview:
        print("")
        print("=== Selection Prompt Preview ===")
        print(selection.selection_prompt_preview)
    print("")
    print("=== Extraction Prompt Preview ===")
    print(context.extraction_prompt_preview)
    print("")
    print("=== Linking Prompt Preview ===")
    print(context.linking_prompt_preview)


if __name__ == "__main__":
    main()
