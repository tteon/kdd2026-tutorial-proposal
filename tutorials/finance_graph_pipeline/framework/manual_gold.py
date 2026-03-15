from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluation import GoldExtraction


@dataclass(frozen=True)
class ManualGoldExample:
    example_id: str
    category: str
    gold_profile: str
    intent_id: str
    required_answer_slots: tuple[str, ...]
    preferred_answer_evidence: tuple[str, ...]
    gold_extraction: GoldExtraction


def load_manual_gold(path: str | Path | None) -> dict[str, ManualGoldExample]:
    if path is None:
        return {}

    payload = json.loads(Path(path).read_text())
    examples: dict[str, ManualGoldExample] = {}
    for item in payload.get("examples", []):
        examples[item["example_id"]] = ManualGoldExample(
            example_id=item["example_id"],
            category=item["category"],
            gold_profile=item["gold_profile"],
            intent_id=item["intent_id"],
            required_answer_slots=tuple(item.get("required_answer_slots", [])),
            preferred_answer_evidence=tuple(item.get("preferred_answer_evidence", [])),
            gold_extraction=GoldExtraction(
                entities=[
                    (entity["name"], entity["entity_type"])
                    for entity in item.get("gold_entities", [])
                ],
                relations=[
                    (relation["source_name"], relation["relation_type"], relation["target_name"])
                    for relation in item.get("gold_relations", [])
                ],
            ),
        )
    return examples
