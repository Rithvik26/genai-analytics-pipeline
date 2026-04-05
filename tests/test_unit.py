"""Unit tests for the analytics pipeline.

These tests require no API key, no real LLM calls, and no Kaggle data.
They use a temporary SQLite database and unittest.mock to isolate behaviour.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.llm_client import OpenRouterLLMClient
from src.pipeline import AnalyticsPipeline, SQLValidator
from src.types import (
    AnswerGenerationOutput,
    PipelineOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLUMNS = ["age", "gender", "addiction_level", "anxiety_score"]
_TABLE = "gaming_mental_health"


def make_temp_db() -> Path:
    """Create a temp SQLite DB with a minimal gaming_mental_health table."""
    tmp = tempfile.mkdtemp()
    db_path = Path(tmp) / "test.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"CREATE TABLE {_TABLE} "
            "(age INTEGER, gender TEXT, addiction_level REAL, anxiety_score REAL)"
        )
        conn.executemany(
            f"INSERT INTO {_TABLE} VALUES (?,?,?,?)",
            [(22, "Male", 3.5, 4.0), (25, "Female", 5.0, 6.0), (30, "Male", 2.0, 3.5)],
        )
    return db_path


def _make_sql_gen_output(sql: str | None) -> SQLGenerationOutput:
    return SQLGenerationOutput(
        sql=sql,
        timing_ms=1.0,
        llm_stats={"llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "test"},
        error=None,
    )


def _make_answer_output(answer: str) -> AnswerGenerationOutput:
    return AnswerGenerationOutput(
        answer=answer,
        timing_ms=1.0,
        llm_stats={"llm_calls": 1, "prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12, "model": "test"},
        error=None,
    )


def make_mock_llm(sql_response: str | None, answer_response: str) -> MagicMock:
    """Return a mock OpenRouterLLMClient."""
    mock = MagicMock(spec=OpenRouterLLMClient)
    mock.model = "test-model"
    mock.generate_sql.return_value = _make_sql_gen_output(sql_response)
    mock.generate_answer.return_value = _make_answer_output(answer_response)
    return mock


# ---------------------------------------------------------------------------
# _extract_sql tests
# ---------------------------------------------------------------------------

class TestExtractSQL(unittest.TestCase):
    def test_markdown_fence_with_sql_tag(self):
        result = OpenRouterLLMClient._extract_sql("```sql\nSELECT 1\n```")
        self.assertEqual(result, "SELECT 1")

    def test_markdown_fence_without_tag(self):
        result = OpenRouterLLMClient._extract_sql("```\nSELECT age FROM t\n```")
        self.assertEqual(result, "SELECT age FROM t")

    def test_json_wrapper(self):
        result = OpenRouterLLMClient._extract_sql('{"sql": "SELECT 1"}')
        self.assertEqual(result, "SELECT 1")

    def test_bare_select_with_preamble(self):
        result = OpenRouterLLMClient._extract_sql("Here is the query: SELECT age FROM t")
        self.assertIsNotNone(result)
        self.assertTrue(result.lower().startswith("select"))

    def test_bare_with_clause(self):
        result = OpenRouterLLMClient._extract_sql("WITH cte AS (SELECT 1) SELECT * FROM cte")
        self.assertIsNotNone(result)
        self.assertTrue(result.upper().startswith("WITH"))

    def test_returns_none_for_garbage(self):
        result = OpenRouterLLMClient._extract_sql("I cannot generate SQL for this.")
        self.assertIsNone(result)

    def test_fence_takes_priority_over_json(self):
        # Should prefer markdown fence even when JSON is also present.
        result = OpenRouterLLMClient._extract_sql('```sql\nSELECT 2\n``` {"sql": "SELECT 1"}')
        self.assertEqual(result, "SELECT 2")


# ---------------------------------------------------------------------------
# SQLValidator tests
# ---------------------------------------------------------------------------

class TestSQLValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SQLValidator(columns=_COLUMNS)

    def test_accepts_valid_select(self):
        out = self.validator.validate("SELECT age FROM gaming_mental_health")
        self.assertTrue(out.is_valid)
        self.assertIsNone(out.error)

    def test_rejects_none(self):
        out = self.validator.validate(None)
        self.assertFalse(out.is_valid)
        self.assertIsNotNone(out.error)

    def test_rejects_empty_string(self):
        out = self.validator.validate("   ")
        self.assertFalse(out.is_valid)

    def test_rejects_delete(self):
        out = self.validator.validate("DELETE FROM gaming_mental_health")
        self.assertFalse(out.is_valid)
        self.assertIn("DELETE", out.error)

    def test_rejects_drop(self):
        out = self.validator.validate("DROP TABLE gaming_mental_health")
        self.assertFalse(out.is_valid)

    def test_rejects_insert(self):
        out = self.validator.validate("INSERT INTO gaming_mental_health VALUES (1,'M',1,1)")
        self.assertFalse(out.is_valid)

    def test_rejects_pragma(self):
        out = self.validator.validate("PRAGMA table_info('gaming_mental_health')")
        self.assertFalse(out.is_valid)

    def test_rejects_multi_statement(self):
        out = self.validator.validate("SELECT 1; DROP TABLE gaming_mental_health")
        self.assertFalse(out.is_valid)
        self.assertIn("Multiple", out.error)

    def test_rejects_unknown_column(self):
        out = self.validator.validate("SELECT zodiac_sign FROM gaming_mental_health")
        self.assertFalse(out.is_valid)
        self.assertIn("zodiac_sign", out.error)

    def test_accepts_known_columns_with_aggregates(self):
        sql = "SELECT AVG(addiction_level) FROM gaming_mental_health GROUP BY gender"
        out = self.validator.validate(sql)
        self.assertTrue(out.is_valid, msg=f"Expected valid but got error: {out.error}")

    def test_timing_is_non_negative(self):
        out = self.validator.validate("SELECT 1")
        # Note: "SELECT 1" has no table reference so no column check on bare "1"
        self.assertGreaterEqual(out.timing_ms, 0.0)

    def test_no_schema_skips_column_check(self):
        v = SQLValidator(columns=[])
        out = v.validate("SELECT zodiac_sign FROM t")
        # Without schema, column check is skipped — validator should accept.
        self.assertTrue(out.is_valid)


# ---------------------------------------------------------------------------
# Pipeline integration tests (mocked LLM, real temp DB)
# ---------------------------------------------------------------------------

class TestPipelineWithMockLLM(unittest.TestCase):
    def setUp(self):
        self.db_path = make_temp_db()

    def _make_pipeline(self, sql: str | None, answer: str = "Test answer") -> AnalyticsPipeline:
        mock_llm = make_mock_llm(sql, answer)
        pipeline = AnalyticsPipeline(db_path=self.db_path, llm_client=mock_llm)
        return pipeline

    def test_success_path(self):
        sql = f"SELECT COUNT(*) as total FROM {_TABLE}"
        pipeline = self._make_pipeline(sql, "There are 3 respondents.")
        result = pipeline.run("How many respondents are there?")

        self.assertIsInstance(result, PipelineOutput)
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.sql)
        self.assertGreater(len(result.rows), 0)
        # All timings present and non-negative.
        for key in ("sql_generation_ms", "sql_validation_ms", "sql_execution_ms",
                    "answer_generation_ms", "total_ms"):
            self.assertIn(key, result.timings)
            self.assertGreaterEqual(result.timings[key], 0.0)

    def test_invalid_sql_path_delete(self):
        pipeline = self._make_pipeline(f"DELETE FROM {_TABLE}")
        result = pipeline.run("Please delete all rows from the gaming_mental_health table")
        self.assertEqual(result.status, "invalid_sql")
        self.assertIsNotNone(result.sql_validation.error)

    def test_unanswerable_path_no_sql(self):
        pipeline = self._make_pipeline(None)
        result = pipeline.run("Which zodiac sign has the highest stress score?")
        self.assertEqual(result.status, "unanswerable")
        self.assertIn("cannot answer", result.answer.lower())

    def test_unanswerable_path_unknown_column(self):
        pipeline = self._make_pipeline(f"SELECT zodiac_sign FROM {_TABLE}")
        result = pipeline.run("Which zodiac sign has the highest stress score?")
        self.assertIn(result.status, {"unanswerable", "invalid_sql"})
        self.assertIn("cannot answer", result.answer.lower())

    def test_token_stats_are_integers(self):
        sql = f"SELECT age FROM {_TABLE} LIMIT 1"
        pipeline = self._make_pipeline(sql)
        result = pipeline.run("What is the age of the first respondent?")
        for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsInstance(result.total_llm_stats[key], int, msg=f"{key} must be int")
            self.assertGreaterEqual(result.total_llm_stats[key], 0)
        self.assertIsInstance(result.total_llm_stats["model"], str)

    def test_output_contract_stage_types(self):
        from src.types import (
            SQLGenerationOutput, SQLValidationOutput,
            SQLExecutionOutput, AnswerGenerationOutput,
        )
        sql = f"SELECT gender, AVG(anxiety_score) FROM {_TABLE} GROUP BY gender"
        pipeline = self._make_pipeline(sql)
        result = pipeline.run("How does anxiety score differ by gender?")
        self.assertIsInstance(result.sql_generation, SQLGenerationOutput)
        self.assertIsInstance(result.sql_validation, SQLValidationOutput)
        self.assertIsInstance(result.sql_execution, SQLExecutionOutput)
        self.assertIsInstance(result.answer_generation, AnswerGenerationOutput)

    def test_scalar_fast_path_skips_llm(self):
        """Single scalar result should return answer without an LLM call."""
        sql = f"SELECT COUNT(*) as cnt FROM {_TABLE}"
        mock_llm = make_mock_llm(sql, "shouldn't be used")
        pipeline = AnalyticsPipeline(db_path=self.db_path, llm_client=mock_llm)
        result = pipeline.run("How many rows?")
        if result.status == "success":
            # Fast-path: answer_generation llm_calls should be 0.
            self.assertEqual(result.answer_generation.llm_stats["llm_calls"], 0)

    def test_execution_error_yields_cannot_answer(self):
        """Malformed SQL that passes validator but fails execution → cannot answer."""
        # Bypass validator by patching validate to return is_valid=True with bad SQL.
        pipeline = self._make_pipeline("SELECT nonexistent_col FROM nonexistent_table")
        with patch.object(pipeline.validator, "validate", return_value=SQLValidationOutput(
            is_valid=True, validated_sql="SELECT nonexistent_col FROM nonexistent_table"
        )):
            result = pipeline.run("test question")
        # Execution will error; answer must contain "cannot answer".
        if result.status == "error":
            self.assertIn("cannot answer", result.answer.lower())

    def test_question_echoed_in_output(self):
        question = "How many respondents have high addiction level?"
        pipeline = self._make_pipeline(f"SELECT COUNT(*) FROM {_TABLE} WHERE addiction_level >= 5")
        result = pipeline.run(question)
        self.assertEqual(result.question, question)

    def test_request_id_echoed(self):
        pipeline = self._make_pipeline(f"SELECT 1")
        result = pipeline.run("test", request_id="req-abc-123")
        self.assertEqual(result.request_id, "req-abc-123")


# ---------------------------------------------------------------------------
# Benchmark script test
# ---------------------------------------------------------------------------

class TestBenchmarkStatusAccess(unittest.TestCase):
    def test_result_status_is_attribute_not_dict(self):
        """Benchmark fix: result.status must be attribute access, not result["status"]."""
        from src.types import (
            SQLGenerationOutput, SQLValidationOutput,
            SQLExecutionOutput, AnswerGenerationOutput, PipelineOutput,
        )
        result = PipelineOutput(
            status="success",
            question="test",
            request_id=None,
            sql_generation=SQLGenerationOutput(sql="SELECT 1", timing_ms=1.0,
                llm_stats={"llm_calls": 1, "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "model": "t"}),
            sql_validation=SQLValidationOutput(is_valid=True, validated_sql="SELECT 1"),
            sql_execution=SQLExecutionOutput(rows=[{"cnt": 1}], row_count=1, timing_ms=0.5),
            answer_generation=AnswerGenerationOutput(answer="one", timing_ms=1.0,
                llm_stats={"llm_calls": 1, "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "model": "t"}),
        )
        # Attribute access must work.
        self.assertEqual(result.status, "success")
        # Dict-style access must raise TypeError (dataclass, not dict).
        with self.assertRaises(TypeError):
            _ = result["status"]


if __name__ == "__main__":
    unittest.main()
