from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tutorials.finance_graph_pipeline.framework.evaluation import FinanceTutorialEvaluator


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "data" / "sample_finance_dataset.json"

    evaluator = FinanceTutorialEvaluator()
    results = evaluator.evaluate(dataset_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

