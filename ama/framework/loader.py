from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


def resolve_hf_token() -> Optional[str]:
    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HG_TOKEN")


@dataclass
class DatasetSlice:
    dataset_id: str
    split: str
    examples: list[dict[str, Any]]


def load_finder_dataset(dataset_id: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency-driven
        raise RuntimeError(
            "The 'datasets' package is required to load FinDER from Hugging Face."
        ) from exc

    token = resolve_hf_token()
    kwargs: dict[str, Any] = {"split": split}
    if token:
        kwargs["token"] = token

    dataset = load_dataset(dataset_id, **kwargs)
    return [dict(row) for row in dataset]


def balanced_sample(
    examples: list[dict[str, Any]],
    *,
    sample_size: int,
    category_field: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    from collections import defaultdict
    from random import Random

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        category = str(example.get(category_field, "unknown"))
        grouped[category].append(example)

    rng = Random(random_seed)
    for values in grouped.values():
        rng.shuffle(values)

    categories = sorted(grouped)
    if not categories:
        return []

    if sample_size <= len(categories):
        rng.shuffle(categories)
        return [grouped[category][0] for category in categories[:sample_size]]

    target_per_category = sample_size // len(categories)
    sampled: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    for category in categories:
        values = grouped[category]
        sampled.extend(values[:target_per_category])
        leftovers.extend(values[target_per_category:])

    if len(sampled) < sample_size:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: sample_size - len(sampled)])

    return sampled[:sample_size]
