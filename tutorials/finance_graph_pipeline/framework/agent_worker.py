from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .finder_experiment import FinDERAnswerAgent, FinDERExample, FinDERExtractionAgent, FinDERProfileAgent
from .models import FiboProfileDecision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI worker for FinDER profile, extraction, and answer calls.")
    parser.add_argument("--task", choices=("profile", "extraction", "answer"), required=True)
    parser.add_argument("--model-name", default="gpt-4.1-mini")
    parser.add_argument("--answer-context-char-budget", type=int, default=2200)
    parser.add_argument("--answer-reference-sentence-limit", type=int, default=4)
    parser.add_argument("--answer-graph-triple-limit", type=int, default=5)
    parser.add_argument("--answer-graph-snippet-limit", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.load(sys.stdin)
    example = FinDERExample(**payload["example"])

    try:
        if args.task == "profile":
            agent = FinDERProfileAgent(mode="openai", model_name=args.model_name)
            decision = agent.select_profile(example)
            result = {
                "status": "ok",
                "selected_profile": decision.selected_profile,
                "confidence": decision.selection_confidence,
                "rationale": decision.ontology_rationale,
            }
        elif args.task == "extraction":
            decision = FiboProfileDecision(**payload["decision"])
            agent = FinDERExtractionAgent(mode="openai", model_name=args.model_name)
            entities, relations, metadata = agent.extract(example, decision)
            result = {
                "status": "ok",
                "entities": [asdict(entity) for entity in entities],
                "relations": [asdict(relation) for relation in relations],
                "metadata": metadata,
            }
        else:
            question = example.to_question()
            agent = FinDERAnswerAgent(
                mode="openai",
                model_name=args.model_name,
                context_char_budget=args.answer_context_char_budget,
                reference_sentence_limit=args.answer_reference_sentence_limit,
                graph_triple_limit=args.answer_graph_triple_limit,
                graph_snippet_limit=args.answer_graph_snippet_limit,
            )
            answer_bundle = agent.answer(
                question=question,
                context_mode=str(payload["context_mode"]),
                context_bundle=dict(payload["context_bundle"]),
            )
            result = {
                "status": "ok",
                "answer": answer_bundle.answer,
                "confidence": answer_bundle.confidence,
                "quality_notes": answer_bundle.quality_notes,
            }
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
