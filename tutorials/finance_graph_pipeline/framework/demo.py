from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tutorials.finance_graph_pipeline.framework.pipeline import FinanceTutorialPipeline


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "data" / "sample_finance_dataset.json"

    pipeline = FinanceTutorialPipeline()
    documents, questions = pipeline.load_dataset(dataset_path)
    pipeline.index_documents(documents)
    pipeline.analyze_quality(questions)
    answers = pipeline.answer_questions(questions)

    output = {
        "profile_decisions": {
            doc_id: {
                "selected_profile": decision.selected_profile,
                "selection_confidence": decision.selection_confidence,
                "mapping_policy": decision.mapping_policy,
            }
            for doc_id, decision in pipeline.state.profile_decisions.items()
        },
        "global_quality_issues": [
            {
                "issue_type": issue.issue_type,
                "severity": issue.severity,
                "message": issue.message,
                "object_type": issue.object_type,
                "object_id": issue.object_id,
            }
            for issue in pipeline.state.quality_issues
        ],
        "query_support": {
            question_id: {
                "answerable": record.answerable,
                "support_score": record.support_score,
                "missing_requirements": record.missing_requirements,
            }
            for question_id, record in pipeline.state.query_support.items()
        },
        "answers": [
            {
                "question_id": answer.question_id,
                "answer": answer.answer,
                "confidence": answer.confidence,
                "selected_edge_ids": answer.selected_edge_ids,
                "quality_notes": answer.quality_notes,
            }
            for answer in answers
        ],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
