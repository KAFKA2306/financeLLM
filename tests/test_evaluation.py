import json
import tempfile
import unittest
from pathlib import Path

from evaluation import RAGEvaluator, evaluate_legacy_csv


FIXTURE = json.loads((Path(__file__).parent / "fixtures/rag_eval/fixture.json").read_text(encoding="utf-8"))


def correct_prediction():
    return {
        "case_id": "revenue-q1",
        "answer": "一次資料によると売上高は120億円です。",
        "retrieved_document_ids": ["annual-report-2026"],
        "citations": [{"document_id": "annual-report-2026", "page": 12, "chunk_id": "p12-c3"}],
        "financial_facts": [{"value": "120", "currency": "JPY", "unit": "億円", "fiscal_year": "2026", "quarter": "Q1"}],
    }


class RAGEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator()
        self.case = FIXTURE["cases"][0]

    def test_correct_answer_is_perfect(self):
        result = self.evaluator.evaluate_case(self.case, correct_prediction())
        self.assertEqual(result["rating"], "Perfect")

    def test_long_wrong_answer_is_not_perfect(self):
        prediction = correct_prediction()
        prediction["answer"] = "誤った説明です。" * 30
        prediction["financial_facts"][0]["value"] = "999"
        result = self.evaluator.evaluate_case(self.case, prediction)
        self.assertEqual(result["rating"], "Incorrect")

    def test_currency_unit_period_mismatch_is_detected(self):
        for field, wrong in (("currency", "USD"), ("unit", "百万円"), ("fiscal_year", "2025"), ("quarter", "Q2")):
            with self.subTest(field=field):
                prediction = correct_prediction()
                prediction["financial_facts"][0][field] = wrong
                result = self.evaluator.evaluate_case(self.case, prediction)
                self.assertFalse(result["metrics"]["numerical_accuracy"]["exact"])
                self.assertNotEqual(result["rating"], "Perfect")

    def test_wrong_citation_is_detected(self):
        prediction = correct_prediction()
        prediction["citations"][0]["chunk_id"] = "p12-c9"
        result = self.evaluator.evaluate_case(self.case, prediction)
        self.assertFalse(result["metrics"]["citation"]["exact"])
        self.assertNotEqual(result["rating"], "Perfect")

    def test_abstention_without_evidence_is_success(self):
        case = FIXTURE["cases"][1]
        result = self.evaluator.evaluate_case(case, {"answer": "根拠を確認できません。", "abstained": True})
        self.assertEqual(result["rating"], "Perfect")

    def test_fixture_output_is_deterministic(self):
        predictions = {"model_id": "fixture-model", "predictions": [correct_prediction(), {"case_id": "no-evidence", "answer": "根拠を確認できません。", "abstained": True}]}
        first = self.evaluator.evaluate_fixture(FIXTURE, predictions)
        second = self.evaluator.evaluate_fixture(FIXTURE, predictions)
        self.assertEqual(first, second)

    def test_legacy_csv_compares_ground_truth_not_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            truth = root / "truth.csv"
            predictions.write_text("1," + "誤答" * 40 + "\n", encoding="utf-8")
            truth.write_text("index,ground_truth\n1,正解\n", encoding="utf-8")
            result = evaluate_legacy_csv(predictions, truth)
            self.assertEqual(result["results"][0]["rating"], "Incorrect")


if __name__ == "__main__":
    unittest.main()
