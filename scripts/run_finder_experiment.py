from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tutorials.finance_graph_pipeline.framework.finder_experiment import parse_args, run_from_args


def main() -> None:
    args = parse_args()
    result = run_from_args(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

