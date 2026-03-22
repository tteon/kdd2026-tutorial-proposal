from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from math import erfc, sqrt
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency-driven
    def load_dotenv() -> bool:
        return False

from framework.loader import load_finder_dataset


DEFAULT_DATASET_ID = "Linq-AI-Research/FinDER"
DEFAULT_SPLIT = "train"
LEXICAL_FIELDS = ("text", "references", "answer")
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "before",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "december",
    "did",
    "do",
    "does",
    "done",
    "during",
    "each",
    "eight",
    "except",
    "five",
    "for",
    "form",
    "four",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "inc",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "ltd",
    "may",
    "million",
    "millions",
    "more",
    "most",
    "net",
    "nine",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "per",
    "plc",
    "same",
    "seven",
    "share",
    "shares",
    "should",
    "since",
    "six",
    "so",
    "some",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "three",
    "through",
    "to",
    "two",
    "under",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "within",
    "would",
    "year",
    "years",
    "you",
    "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze category-level lexical repetition and reasoning overlap in FinDER."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--output",
        default="docs/finder_category_signal_analysis.json",
        help="Path to write the JSON summary.",
    )
    return parser.parse_args()


def join_field_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&./%-]*", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        normalized = token.strip("./-%$()—–-,:;[]{}")
        if len(normalized) < 3:
            continue
        if normalized in STOPWORDS:
            continue
        if normalized.isdigit():
            continue
        tokens.append(normalized)
    return tokens


def phrase_stopword(token: str) -> bool:
    return token in STOPWORDS


def extract_phrase_candidates(
    text: str,
    *,
    min_len: int = 2,
    max_len: int = 4,
    boilerplate_terms: set[str] | None = None,
) -> list[str]:
    tokens = tokenize(text)
    phrases: set[str] = set()
    for start in range(len(tokens)):
        if phrase_stopword(tokens[start]):
            continue
        for width in range(min_len, max_len + 1):
            end = start + width
            if end > len(tokens):
                break
            window = tokens[start:end]
            if phrase_stopword(window[-1]):
                continue
            if boilerplate_terms and any(token in boilerplate_terms for token in window):
                continue
            if len(set(window)) == 1:
                continue
            if any(len(token) < 3 for token in window):
                continue
            phrase = " ".join(window)
            phrases.add(phrase)
    return sorted(phrases)


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def summarize_field(
    examples: list[dict[str, Any]],
    *,
    field_name: str,
    category_field: str = "category",
) -> dict[str, dict[str, Any]]:
    category_tokens: dict[str, list[str]] = defaultdict(list)
    category_doc_freq: dict[str, Counter[str]] = defaultdict(Counter)
    category_examples: Counter[str] = Counter()

    for example in examples:
        category = str(example.get(category_field, "unknown"))
        joined = join_field_value(example.get(field_name, ""))
        tokens = tokenize(joined)
        category_examples[category] += 1
        category_tokens[category].extend(tokens)
        for token in set(tokens):
            category_doc_freq[category][token] += 1

    summary: dict[str, dict[str, Any]] = {}
    for category in sorted(category_tokens):
        counts = Counter(category_tokens[category])
        duplicates = {token: count for token, count in counts.items() if count > 1}
        duplicate_mentions = sum(count - 1 for count in duplicates.values())
        top_by_document = sorted(
            category_doc_freq[category].items(),
            key=lambda item: (-item[1], item[0]),
        )[:15]
        summary[category] = {
            "examples": category_examples[category],
            "total_tokens": sum(counts.values()),
            "unique_tokens": len(counts),
            "duplicate_mentions": duplicate_mentions,
            "duplicate_rate": ratio(duplicate_mentions, sum(counts.values())),
            "top_document_frequency_terms": [
                {"term": term, "document_frequency": freq} for term, freq in top_by_document
            ],
        }
    return summary


def collect_document_frequencies(
    examples: list[dict[str, Any]],
    *,
    field_name: str,
    category_field: str = "category",
    boilerplate_terms: set[str] | None = None,
) -> tuple[dict[str, Counter[str]], Counter[str], Counter[str]]:
    category_doc_freq: dict[str, Counter[str]] = defaultdict(Counter)
    category_examples: Counter[str] = Counter()
    global_doc_freq: Counter[str] = Counter()

    for example in examples:
        category = str(example.get(category_field, "unknown"))
        tokens = set(tokenize(join_field_value(example.get(field_name, ""))))
        if boilerplate_terms:
            tokens = {token for token in tokens if token not in boilerplate_terms}
        category_examples[category] += 1
        for token in tokens:
            category_doc_freq[category][token] += 1
            global_doc_freq[token] += 1

    return category_doc_freq, category_examples, global_doc_freq


def infer_reference_boilerplate(
    examples: list[dict[str, Any]],
    *,
    minimum_document_rate: float = 0.2,
) -> list[dict[str, Any]]:
    category_doc_freq, category_examples, global_doc_freq = collect_document_frequencies(
        examples,
        field_name="references",
    )
    total_examples = sum(category_examples.values())
    category_count = len(category_examples)
    boilerplate: list[dict[str, Any]] = []

    for term, document_frequency in global_doc_freq.items():
        global_rate = document_frequency / total_examples if total_examples else 0.0
        categories_present = sum(
            1 for category in category_doc_freq if category_doc_freq[category].get(term, 0) > 0
        )
        if global_rate >= minimum_document_rate and categories_present >= max(3, category_count // 2):
            boilerplate.append(
                {
                    "term": term,
                    "document_frequency": document_frequency,
                    "global_document_rate": round(global_rate, 4),
                    "categories_present": categories_present,
                }
            )

    boilerplate.sort(
        key=lambda item: (
            -item["global_document_rate"],
            -item["categories_present"],
            item["term"],
        )
    )
    return boilerplate[:50]


def chi_square_p_value(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    total = a + b + c + d
    if total == 0:
        return 0.0, 1.0
    numerator = total * ((a * d) - (b * c)) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator == 0:
        return 0.0, 1.0
    chi_square = numerator / denominator
    p_value = erfc(sqrt(chi_square / 2.0))
    return chi_square, p_value


def summarize_category_specific_terms(
    examples: list[dict[str, Any]],
    *,
    field_name: str = "references",
    top_k: int = 20,
    min_category_doc_frequency: int = 20,
    min_rate_gap: float = 0.05,
    max_p_value: float = 0.001,
    boilerplate_terms: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    category_doc_freq, category_examples, global_doc_freq = collect_document_frequencies(
        examples,
        field_name=field_name,
        boilerplate_terms=boilerplate_terms,
    )
    total_examples = sum(category_examples.values())
    summary: dict[str, list[dict[str, Any]]] = {}

    for category, doc_freq in sorted(category_doc_freq.items()):
        category_total = category_examples[category]
        outside_total = total_examples - category_total
        significant_terms: list[dict[str, Any]] = []
        for term, in_category in doc_freq.items():
            if in_category < min_category_doc_frequency:
                continue
            outside_count = global_doc_freq[term] - in_category
            in_rate = in_category / category_total if category_total else 0.0
            out_rate = outside_count / outside_total if outside_total else 0.0
            rate_gap = in_rate - out_rate
            if rate_gap < min_rate_gap:
                continue
            a = in_category
            b = category_total - in_category
            c = outside_count
            d = outside_total - outside_count
            chi_square, p_value = chi_square_p_value(a, b, c, d)
            if p_value > max_p_value:
                continue
            significant_terms.append(
                {
                    "term": term,
                    "category_document_frequency": in_category,
                    "category_document_rate": round(in_rate, 4),
                    "outside_document_rate": round(out_rate, 4),
                    "rate_gap": round(rate_gap, 4),
                    "chi_square": round(chi_square, 4),
                    "p_value": p_value,
                }
            )

        significant_terms.sort(
            key=lambda item: (
                item["p_value"],
                -item["rate_gap"],
                -item["category_document_frequency"],
                item["term"],
            )
        )
        summary[category] = significant_terms[:top_k]

    return summary


def summarize_duplicate_rankings(
    field_summary: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for field_name, per_category in field_summary.items():
        rankings[field_name] = [
            {
                "category": category,
                "duplicate_rate": stats["duplicate_rate"],
                "total_tokens": stats["total_tokens"],
            }
            for category, stats in sorted(
                per_category.items(),
                key=lambda item: (-item[1]["duplicate_rate"], item[0]),
            )
        ]
    return rankings


def collect_reference_token_sequences(
    examples: list[dict[str, Any]],
    *,
    boilerplate_terms: set[str] | None = None,
) -> dict[str, list[list[str]]]:
    sequences: dict[str, list[list[str]]] = defaultdict(list)
    for example in examples:
        category = str(example.get("category", "unknown"))
        tokens = tokenize(join_field_value(example.get("references", "")))
        if boilerplate_terms:
            tokens = [token for token in tokens if token not in boilerplate_terms]
        if tokens:
            sequences[category].append(tokens)
    return sequences


def summarize_shared_term_contexts(
    examples: list[dict[str, Any]],
    *,
    boilerplate_terms: set[str] | None = None,
    min_categories: int = 3,
    top_terms: int = 8,
    context_window: int = 2,
) -> list[dict[str, Any]]:
    category_doc_freq, category_examples, _ = collect_document_frequencies(
        examples,
        field_name="references",
        boilerplate_terms=boilerplate_terms,
    )
    sequences = collect_reference_token_sequences(examples, boilerplate_terms=boilerplate_terms)
    category_presence: Counter[str] = Counter()
    total_doc_freq: Counter[str] = Counter()

    for category, doc_freq in category_doc_freq.items():
        for term, freq in doc_freq.items():
            if freq > 0:
                category_presence[term] += 1
                total_doc_freq[term] += freq

    candidates = [
        term
        for term, presence in category_presence.items()
        if presence >= min_categories and total_doc_freq[term] >= 80
    ]
    candidates.sort(key=lambda term: (-total_doc_freq[term], term))

    summaries: list[dict[str, Any]] = []
    for term in candidates[:top_terms]:
        category_contexts: list[dict[str, Any]] = []
        for category, token_lists in sequences.items():
            if category_doc_freq[category].get(term, 0) == 0:
                continue
            neighbors: Counter[str] = Counter()
            for tokens in token_lists:
                for index, token in enumerate(tokens):
                    if token != term:
                        continue
                    start = max(0, index - context_window)
                    end = min(len(tokens), index + context_window + 1)
                    for neighbor in tokens[start:end]:
                        if neighbor != term:
                            neighbors[neighbor] += 1
            category_contexts.append(
                {
                    "category": category,
                    "document_rate": round(
                        category_doc_freq[category][term] / category_examples[category], 4
                    ),
                    "top_neighbors": [
                        {"term": neighbor, "count": count}
                        for neighbor, count in neighbors.most_common(5)
                    ],
                }
            )

        category_contexts.sort(key=lambda item: (-item["document_rate"], item["category"]))
        summaries.append(
            {
                "term": term,
                "category_count": category_presence[term],
                "total_document_frequency": total_doc_freq[term],
                "contexts": category_contexts[:5],
            }
        )

    return summaries


def collect_phrase_document_frequencies(
    examples: list[dict[str, Any]],
    *,
    field_name: str = "references",
    category_field: str = "category",
    boilerplate_terms: set[str] | None = None,
) -> tuple[dict[str, Counter[str]], Counter[str], Counter[str]]:
    category_doc_freq: dict[str, Counter[str]] = defaultdict(Counter)
    category_examples: Counter[str] = Counter()
    global_doc_freq: Counter[str] = Counter()

    for example in examples:
        category = str(example.get(category_field, "unknown"))
        phrases = extract_phrase_candidates(
            join_field_value(example.get(field_name, "")),
            boilerplate_terms=boilerplate_terms,
        )
        category_examples[category] += 1
        for phrase in set(phrases):
            category_doc_freq[category][phrase] += 1
            global_doc_freq[phrase] += 1

    return category_doc_freq, category_examples, global_doc_freq


def summarize_category_specific_phrases(
    examples: list[dict[str, Any]],
    *,
    field_name: str = "references",
    top_k: int = 15,
    min_category_doc_frequency: int = 12,
    min_rate_gap: float = 0.03,
    max_p_value: float = 0.001,
    boilerplate_terms: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    category_doc_freq, category_examples, global_doc_freq = collect_phrase_document_frequencies(
        examples,
        field_name=field_name,
        boilerplate_terms=boilerplate_terms,
    )
    total_examples = sum(category_examples.values())
    summary: dict[str, list[dict[str, Any]]] = {}

    for category, doc_freq in sorted(category_doc_freq.items()):
        category_total = category_examples[category]
        outside_total = total_examples - category_total
        significant_phrases: list[dict[str, Any]] = []
        for phrase, in_category in doc_freq.items():
            if in_category < min_category_doc_frequency:
                continue
            outside_count = global_doc_freq[phrase] - in_category
            in_rate = in_category / category_total if category_total else 0.0
            out_rate = outside_count / outside_total if outside_total else 0.0
            rate_gap = in_rate - out_rate
            if rate_gap < min_rate_gap:
                continue
            a = in_category
            b = category_total - in_category
            c = outside_count
            d = outside_total - outside_count
            chi_square, p_value = chi_square_p_value(a, b, c, d)
            if p_value > max_p_value:
                continue
            significant_phrases.append(
                {
                    "phrase": phrase,
                    "category_document_frequency": in_category,
                    "category_document_rate": round(in_rate, 4),
                    "outside_document_rate": round(out_rate, 4),
                    "rate_gap": round(rate_gap, 4),
                    "chi_square": round(chi_square, 4),
                    "p_value": p_value,
                }
            )

        significant_phrases.sort(
            key=lambda item: (
                item["p_value"],
                -item["rate_gap"],
                -item["category_document_frequency"],
                item["phrase"],
            )
        )
        summary[category] = significant_phrases[:top_k]

    return summary


def summarize_shared_phrases(
    examples: list[dict[str, Any]],
    *,
    field_name: str = "references",
    boilerplate_terms: set[str] | None = None,
    min_categories: int = 2,
    min_total_document_frequency: int = 20,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    category_doc_freq, category_examples, global_doc_freq = collect_phrase_document_frequencies(
        examples,
        field_name=field_name,
        boilerplate_terms=boilerplate_terms,
    )
    phrase_category_presence: Counter[str] = Counter()
    for category, doc_freq in category_doc_freq.items():
        for phrase, freq in doc_freq.items():
            if freq > 0:
                phrase_category_presence[phrase] += 1

    shared = [
        {
            "phrase": phrase,
            "category_count": phrase_category_presence[phrase],
            "total_document_frequency": global_doc_freq[phrase],
            "top_categories": [
                {
                    "category": category,
                    "document_rate": round(
                        category_doc_freq[category].get(phrase, 0) / category_examples[category], 4
                    ),
                }
                for category in sorted(
                    category_doc_freq,
                    key=lambda category: (
                        -category_doc_freq[category].get(phrase, 0) / category_examples[category],
                        category,
                    ),
                )
                if category_doc_freq[category].get(phrase, 0) > 0
            ][:4],
        }
        for phrase in global_doc_freq
        if phrase_category_presence[phrase] >= min_categories
        and global_doc_freq[phrase] >= min_total_document_frequency
    ]
    shared.sort(
        key=lambda item: (-item["total_document_frequency"], -item["category_count"], item["phrase"])
    )
    return shared[:top_k]


def summarize_reasoning_flags(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "examples": 0,
            "reasoning_true": 0,
            "reference_tokens_total": 0,
            "answer_tokens_total": 0,
            "text_tokens_total": 0,
            "reference_tokens_reasoning_true": 0,
            "reference_tokens_reasoning_false": 0,
            "answer_tokens_reasoning_true": 0,
            "answer_tokens_reasoning_false": 0,
        }
    )

    for example in examples:
        category = str(example.get("category", "unknown"))
        reasoning_value = bool(example.get("reasoning"))
        reference_tokens = tokenize(join_field_value(example.get("references", "")))
        answer_tokens = tokenize(join_field_value(example.get("answer", "")))
        text_tokens = tokenize(join_field_value(example.get("text", "")))

        metrics = per_category[category]
        metrics["examples"] += 1
        metrics["reasoning_true"] += 1 if reasoning_value else 0
        metrics["reference_tokens_total"] += len(reference_tokens)
        metrics["answer_tokens_total"] += len(answer_tokens)
        metrics["text_tokens_total"] += len(text_tokens)

        if reasoning_value:
            metrics["reference_tokens_reasoning_true"] += len(reference_tokens)
            metrics["answer_tokens_reasoning_true"] += len(answer_tokens)
        else:
            metrics["reference_tokens_reasoning_false"] += len(reference_tokens)
            metrics["answer_tokens_reasoning_false"] += len(answer_tokens)

    summary: dict[str, dict[str, Any]] = {}
    for category, metrics in sorted(per_category.items()):
        examples_count = metrics["examples"]
        reasoning_true = metrics["reasoning_true"]
        reasoning_false = examples_count - reasoning_true
        summary[category] = {
            "examples": examples_count,
            "reasoning_true": reasoning_true,
            "reasoning_false": reasoning_false,
            "reasoning_true_rate": ratio(reasoning_true, examples_count),
            "avg_reference_tokens": round(metrics["reference_tokens_total"] / examples_count, 2)
            if examples_count
            else 0.0,
            "avg_answer_tokens": round(metrics["answer_tokens_total"] / examples_count, 2)
            if examples_count
            else 0.0,
            "avg_text_tokens": round(metrics["text_tokens_total"] / examples_count, 2)
            if examples_count
            else 0.0,
            "avg_reference_tokens_when_reasoning_true": round(
                metrics["reference_tokens_reasoning_true"] / reasoning_true, 2
            )
            if reasoning_true
            else 0.0,
            "avg_reference_tokens_when_reasoning_false": round(
                metrics["reference_tokens_reasoning_false"] / reasoning_false, 2
            )
            if reasoning_false
            else 0.0,
            "avg_answer_tokens_when_reasoning_true": round(
                metrics["answer_tokens_reasoning_true"] / reasoning_true, 2
            )
            if reasoning_true
            else 0.0,
            "avg_answer_tokens_when_reasoning_false": round(
                metrics["answer_tokens_reasoning_false"] / reasoning_false, 2
            )
            if reasoning_false
            else 0.0,
        }
    return summary


def make_comparison(
    field_summary: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    categories = sorted(field_summary["references"])
    for category in categories:
        text_stats = field_summary["text"][category]
        reference_stats = field_summary["references"][category]
        answer_stats = field_summary["answer"][category]
        comparison[category] = {
            "text_duplicate_rate": text_stats["duplicate_rate"],
            "references_duplicate_rate": reference_stats["duplicate_rate"],
            "answer_duplicate_rate": answer_stats["duplicate_rate"],
            "references_minus_text_duplicate_rate": round(
                reference_stats["duplicate_rate"] - text_stats["duplicate_rate"], 4
            ),
            "answer_minus_text_duplicate_rate": round(
                answer_stats["duplicate_rate"] - text_stats["duplicate_rate"], 4
            ),
        }
    return comparison


def build_summary(examples: list[dict[str, Any]], *, dataset_id: str, split: str) -> dict[str, Any]:
    reference_boilerplate = infer_reference_boilerplate(examples)
    boilerplate_terms = {item["term"] for item in reference_boilerplate}
    field_summary = {
        field_name: summarize_field(examples, field_name=field_name)
        for field_name in LEXICAL_FIELDS
    }
    cleaned_reference_summary = summarize_field(
        examples,
        field_name="references",
    )
    if boilerplate_terms:
        cleaned_reference_summary = {}
        category_doc_freq, category_examples, _ = collect_document_frequencies(
            examples,
            field_name="references",
            boilerplate_terms=boilerplate_terms,
        )
        category_tokens: dict[str, list[str]] = defaultdict(list)
        for example in examples:
            category = str(example.get("category", "unknown"))
            tokens = [
                token
                for token in tokenize(join_field_value(example.get("references", "")))
                if token not in boilerplate_terms
            ]
            category_tokens[category].extend(tokens)
        for category in sorted(category_tokens):
            counts = Counter(category_tokens[category])
            duplicates = {token: count for token, count in counts.items() if count > 1}
            duplicate_mentions = sum(count - 1 for count in duplicates.values())
            cleaned_reference_summary[category] = {
                "examples": category_examples[category],
                "total_tokens": sum(counts.values()),
                "unique_tokens": len(counts),
                "duplicate_mentions": duplicate_mentions,
                "duplicate_rate": ratio(duplicate_mentions, sum(counts.values())),
                "top_document_frequency_terms": [
                    {"term": term, "document_frequency": freq}
                    for term, freq in sorted(
                        category_doc_freq[category].items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:15]
                ],
            }
    return {
        "dataset_id": dataset_id,
        "split": split,
        "example_count": len(examples),
        "fields": field_summary,
        "reference_boilerplate_terms": reference_boilerplate,
        "cleaned_references": cleaned_reference_summary,
        "comparisons": make_comparison(field_summary),
        "duplicate_rankings": summarize_duplicate_rankings(field_summary),
        "category_specific_reference_terms": summarize_category_specific_terms(
            examples,
            field_name="references",
            boilerplate_terms=boilerplate_terms,
        ),
        "category_specific_reference_phrases": summarize_category_specific_phrases(
            examples,
            field_name="references",
            boilerplate_terms=boilerplate_terms,
        ),
        "shared_reference_term_contexts": summarize_shared_term_contexts(
            examples,
            boilerplate_terms=boilerplate_terms,
        ),
        "shared_reference_phrases": summarize_shared_phrases(
            examples,
            field_name="references",
            boilerplate_terms=boilerplate_terms,
        ),
        "reasoning_flags": summarize_reasoning_flags(examples),
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    examples = load_finder_dataset(args.dataset_id, args.split)
    summary = build_summary(examples, dataset_id=args.dataset_id, split=args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
