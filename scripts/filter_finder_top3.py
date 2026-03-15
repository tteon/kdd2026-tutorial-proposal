from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TARGET_CATEGORIES = (
    "Governance",
    "Financials",
    "Shareholder Return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter the FinDER parquet file down to the ontology-sensitive top-3 categories."
    )
    parser.add_argument(
        "--input",
        default="/workspace/dataset/train-00000-of-00001.parquet",
        help="Path to the FinDER parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/app/data/processed/finder_top3",
        help="Directory where the filtered outputs will be written.",
    )
    return parser.parse_args()


def normalize_filename(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def normalize_category(value: str) -> str:
    return " ".join(value.lower().split())


def category_counts(table: pa.Table) -> dict[str, int]:
    return dict(sorted(Counter(table["category"].to_pylist()).items()))


def filter_categories(table: pa.Table, categories: tuple[str, ...]) -> pa.Table:
    normalized_targets = [normalize_category(category) for category in categories]
    normalized_column = pc.utf8_lower(table["category"])
    mask = pc.is_in(normalized_column, value_set=pa.array(normalized_targets))
    return table.filter(mask)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(input_path)
    schema_names = set(table.schema.names)
    required_columns = {"_id", "text", "reasoning", "category", "references", "answer", "type"}
    missing_columns = sorted(required_columns - schema_names)
    if missing_columns:
        raise ValueError(
            f"FinDER schema does not match expectations. Missing columns: {', '.join(missing_columns)}"
        )

    full_counts = category_counts(table)
    subset = filter_categories(table, TARGET_CATEGORIES)
    subset_counts = category_counts(subset)

    pq.write_table(subset, output_dir / "finder_top3.parquet")

    for category in TARGET_CATEGORIES:
        category_subset = filter_categories(subset, (category,))
        pq.write_table(category_subset, output_dir / f"{normalize_filename(category)}.parquet")

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "input_rows": table.num_rows,
        "output_rows": subset.num_rows,
        "target_categories": list(TARGET_CATEGORIES),
        "all_category_counts": full_counts,
        "selected_category_counts": subset_counts,
        "schema": [{"name": field.name, "type": str(field.type)} for field in table.schema],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
