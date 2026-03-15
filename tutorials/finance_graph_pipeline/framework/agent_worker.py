from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .finder_experiment import FinDERExample, FinDERExtractionAgent, FinDERProfileAgent
from .models import FiboProfileDecision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI worker for FinDER profile/extraction calls.")
    parser.add_argument("--task", choices=("profile", "extraction"), required=True)
    parser.add_argument("--model-name", default="gpt-4.1-mini")
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
        else:
            decision = FiboProfileDecision(**payload["decision"])
            agent = FinDERExtractionAgent(mode="openai", model_name=args.model_name)
            entities, relations, metadata = agent.extract(example, decision)
            result = {
                "status": "ok",
                "entities": [asdict(entity) for entity in entities],
                "relations": [asdict(relation) for relation in relations],
                "metadata": metadata,
            }
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
