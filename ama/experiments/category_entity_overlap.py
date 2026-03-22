from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from framework.analysis import category_duplicate_entity_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize duplicate linked entities by category from artifact JSON files."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--representation", choices=("rdf", "lpg"), default=None)
    return parser.parse_args()


def load_artifacts(base_dir: Path, representation: Optional[str]) -> list[dict[str, Any]]:
    if representation:
        candidates = sorted((base_dir / representation).glob("*.json"))
    else:
        candidates = sorted(base_dir.glob("**/*.json"))

    artifacts: list[dict[str, Any]] = []
    for path in candidates:
        if path.name == "manifest.json":
            continue
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    return artifacts


def main() -> None:
    args = parse_args()
    base_dir = Path(args.artifact_dir)
    artifacts = load_artifacts(base_dir, args.representation)
    summary = category_duplicate_entity_summary(artifacts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
