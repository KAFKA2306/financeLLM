"""Deterministic, offline evaluation for finance RAG answers."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

RATING_SCORES = {"Perfect": 1.0, "Acceptable": 0.5, "Missing": 0.0, "Incorrect": -1.0}
ABSTENTION_MARKERS = (
    "根拠がありません",
    "根拠を確認できません",
    "確認できません",
    "回答できません",
    "insufficient evidence",
    "cannot answer",
)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _citation_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _norm(item.get("document_id")),
        _norm(item.get("page")),
        _norm(item.get("chunk_id")),
    )


def _fact_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _norm(item.get("value")),
        _norm(item.get("currency")),
        _norm(item.get("unit")),
        _norm(item.get("fiscal_year")),
        _norm(item.get("quarter")),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    fixture_revision: str = "rag-eval-v1"
    evaluator_version: str = "2.0.0"


class RAGEvaluator:
    """Evaluate retrieval, citations, grounding, financial facts and abstention."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()

    def evaluate_case(self, case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        answer = str(prediction.get("answer", "")).strip()
        requires_abstention = bool(case.get("requires_abstention", False))

        expected_docs = [str(x) for x in case.get("expected_document_ids", [])]
        retrieved_docs = [str(x) for x in prediction.get("retrieved_document_ids", [])]
        retrieved_norm = [_norm(x) for x in retrieved_docs]
        expected_norm = [_norm(x) for x in expected_docs]
        hits = [doc for doc in expected_norm if doc in retrieved_norm]
        ranks = [retrieved_norm.index(doc) + 1 for doc in hits]
        retrieval = {
            "hit": len(hits) == len(expected_norm),
            "hit_count": len(hits),
            "expected_count": len(expected_norm),
            "best_rank": min(ranks) if ranks else None,
        }

        expected_citations = {_citation_key(x) for x in case.get("expected_citations", [])}
        actual_citations = {_citation_key(x) for x in prediction.get("citations", [])}
        citation_matches = expected_citations & actual_citations
        citation = {
            "precision": _safe_ratio(len(citation_matches), len(actual_citations)),
            "recall": _safe_ratio(len(citation_matches), len(expected_citations)),
            "exact": expected_citations == actual_citations,
        }

        expected_facts = {_fact_key(x) for x in case.get("expected_financial_facts", [])}
        actual_facts = {_fact_key(x) for x in prediction.get("financial_facts", [])}
        fact_matches = expected_facts & actual_facts
        numerical = {
            "precision": _safe_ratio(len(fact_matches), len(actual_facts)),
            "recall": _safe_ratio(len(fact_matches), len(expected_facts)),
            "exact": expected_facts == actual_facts,
        }

        required_phrases = [_norm(x) for x in case.get("required_answer_phrases", [])]
        normalized_answer = _norm(answer)
        groundedness = {
            "required_phrase_recall": _safe_ratio(
                sum(phrase in normalized_answer for phrase in required_phrases),
                len(required_phrases),
            ),
            "all_required_phrases_present": all(
                phrase in normalized_answer for phrase in required_phrases
            ),
        }

        explicit_abstention = bool(prediction.get("abstained", False)) or any(
            marker in normalized_answer for marker in ABSTENTION_MARKERS
        )
        abstention = {
            "required": requires_abstention,
            "observed": explicit_abstention,
            "correct": explicit_abstention if requires_abstention else not explicit_abstention,
        }

        if not answer and not explicit_abstention:
            rating = "Missing"
        elif requires_abstention:
            rating = "Perfect" if abstention["correct"] and not actual_facts else "Incorrect"
        else:
            critical_ok = (
                retrieval["hit"]
                and citation["exact"]
                and numerical["exact"]
                and groundedness["all_required_phrases_present"]
                and abstention["correct"]
            )
            if critical_ok:
                rating = "Perfect"
            elif numerical["exact"] and groundedness["all_required_phrases_present"]:
                rating = "Acceptable"
            else:
                rating = "Incorrect"

        return {
            "case_id": case.get("case_id"),
            "rating": rating,
            "score": RATING_SCORES[rating],
            "metrics": {
                "retrieval": retrieval,
                "citation": citation,
                "groundedness": groundedness,
                "numerical_accuracy": numerical,
                "abstention": abstention,
            },
        }

    def evaluate_fixture(self, fixture: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
        prediction_by_id = {str(x["case_id"]): x for x in predictions.get("predictions", [])}
        results = []
        for case in fixture.get("cases", []):
            case_id = str(case["case_id"])
            results.append(self.evaluate_case(case, prediction_by_id.get(case_id, {})))

        return {
            "fixture_revision": fixture.get("fixture_revision", self.config.fixture_revision),
            "evaluator_version": self.config.evaluator_version,
            "model_id": predictions.get("model_id", "unknown"),
            "case_count": len(results),
            "average_score": sum(x["score"] for x in results) / len(results) if results else 0.0,
            "results": results,
        }


def evaluate_legacy_csv(predictions_path: Path, ground_truth_path: Path) -> dict[str, Any]:
    """Compare legacy free-text responses with ground-truth text instead of response length."""
    with predictions_path.open(encoding="utf-8", newline="") as handle:
        predictions = {str(row[0]): row[1] for row in csv.reader(handle) if len(row) >= 2}
    with ground_truth_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = []
    for row in rows:
        index = str(row["index"])
        expected = _norm(row.get("ground_truth"))
        actual = _norm(predictions.get(index, ""))
        if not actual:
            rating = "Missing"
        elif expected and expected in actual:
            rating = "Perfect"
        else:
            rating = "Incorrect"
        results.append({"index": index, "rating": rating, "score": RATING_SCORES[rating]})
    return {"case_count": len(results), "results": results}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.fixture:
        result = RAGEvaluator().evaluate_fixture(_load_json(args.fixture), _load_json(args.predictions))
    elif args.ground_truth:
        result = evaluate_legacy_csv(args.predictions, args.ground_truth)
    else:
        parser.error("--fixture or --ground-truth is required")

    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
