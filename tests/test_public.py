from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.pipeline import AnalyticsPipeline
from scripts.gaming_csv_to_db import csv_to_sqlite
from scripts.gaming_csv_to_db import DEFAULT_CSV_PATH, DEFAULT_DB_PATH, DEFAULT_TABLE_NAME
from src.types import (
    AnswerGenerationOutput,
    PipelineOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)


def _ensure_gaming_db() -> Path:
    """Ensure gaming mental health DB exists; create from CSV if missing."""
    if not DEFAULT_DB_PATH.exists():
        csv_to_sqlite(DEFAULT_CSV_PATH, DEFAULT_DB_PATH, DEFAULT_TABLE_NAME, if_exists="replace")
    return DEFAULT_DB_PATH


@unittest.skipUnless(os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY is required for LLM integration tests.")
class PublicPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        db_path = _ensure_gaming_db()
        cls.pipeline = AnalyticsPipeline(db_path=db_path)

    def _assert_internal_eval_contract(self, result: PipelineOutput, expected_question: str) -> None:
        self.assertIsInstance(result, PipelineOutput)
        self.assertIn(result.status, {"success", "unanswerable", "invalid_sql", "error"})
        self.assertEqual(result.question, expected_question)

        # Internal eval expects strongly typed stage outputs.
        self.assertIsInstance(result.sql_generation, SQLGenerationOutput)
        self.assertIsInstance(result.sql_validation, SQLValidationOutput)
        self.assertIsInstance(result.sql_execution, SQLExecutionOutput)
        self.assertIsInstance(result.answer_generation, AnswerGenerationOutput)

        self.assertIsInstance(result.timings, dict)
        for key in (
            "sql_generation_ms",
            "sql_validation_ms",
            "sql_execution_ms",
            "answer_generation_ms",
            "total_ms",
        ):
            self.assertIn(key, result.timings)
            self.assertIsInstance(result.timings[key], (int, float))
            self.assertGreaterEqual(result.timings[key], 0.0)

        # Internal scoring uses total_llm_stats for efficiency metrics.
        self.assertIsInstance(result.total_llm_stats, dict)
        for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIn(key, result.total_llm_stats)
            self.assertIsInstance(result.total_llm_stats[key], int)
            self.assertGreaterEqual(result.total_llm_stats[key], 0)
        self.assertIn("model", result.total_llm_stats)
        self.assertIsInstance(result.total_llm_stats["model"], str)

    def test_answerable_prompt_returns_sql_and_answer(self) -> None:
        question = "What are the top 5 age groups by average addiction level?"
        result = self.pipeline.run(question)
        self._assert_internal_eval_contract(result, question)
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.sql)
        self.assertTrue(result.sql.lower().startswith("select"))
        self.assertGreater(len(result.rows), 0)

    def test_unanswerable_prompt_is_handled(self) -> None:
        result = self.pipeline.run("Which zodiac sign has the highest stress score?")
        self.assertIn(result.status, {"unanswerable", "invalid_sql"})
        self.assertIn("cannot answer", result.answer.lower())

    def test_invalid_sql_is_rejected(self) -> None:
        result = self.pipeline.run("Please delete all rows from the gaming_mental_health table")
        self.assertEqual(result.status, "invalid_sql")
        self.assertIsNotNone(result.sql_validation.error)

    def test_timings_exist(self) -> None:
        result = self.pipeline.run("How many respondents have high addiction level (>= 5)?")
        for key in (
            "sql_generation_ms",
            "sql_validation_ms",
            "sql_execution_ms",
            "answer_generation_ms",
            "total_ms",
        ):
            self.assertIn(key, result.timings)
            self.assertGreaterEqual(result.timings[key], 0.0)

    def test_output_contract_is_internal_eval_compatible(self) -> None:
        question = "How does average anxiety score differ by addiction level?"
        result = self.pipeline.run(question)
        self._assert_internal_eval_contract(result, question)


if __name__ == "__main__":
    unittest.main()
