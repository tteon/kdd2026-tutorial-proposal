from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the SQLite metadata database for tutorial experiments.")
    parser.add_argument(
        "--db-path",
        default=os.getenv("METADATA_DB_PATH", "/workspace/app/data/metadata/experiment.sqlite"),
        help="SQLite database file to create or update.",
    )
    parser.add_argument(
        "--schema-path",
        default="/workspace/app/sql/init_metadata.sql",
        help="SQL schema file to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    schema_path = Path(args.schema_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    schema = schema_path.read_text()
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema)

    print(f"Initialized metadata database at {db_path}")


if __name__ == "__main__":
    main()

