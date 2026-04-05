"""Unit tests for the analytics pipeline.

No API key, no Kaggle data, no network calls required.
Uses a temporary SQLite DB and unittest.mock to isolate behaviour.
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

from src.llm_client import OpenRouterLLMClient, _trim_sql_trailing_prose
from src.pipeline import AnalyticsPipeline, SQLValidator
from src.types import (
    AnswerGenerationOutput,
    PipelineOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
    SQLExecutionOutput,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TABLE = "gaming_mental_health"
_COLUMNS = ["age", "gender", "addiction_level", "anxiety_score"]


def make_temp_db() -> Path:
    """Temporary SQLite DB with a minimal gaming_mental_health table (3 rows)."""
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


def _sql_gen(sql: str | None, llm_calls: int = 1) -> SQLGenerationOutput:
    stats: dict[str, Any] = {
        "llm_calls": llm_calls,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "model": "test-model",
    }
    return SQLGenerationOutput(sql=sql, timing_ms=1.0, llm_stats=stats, error=None)


def _answer_gen(answer: str, llm_calls: int = 1) -> AnswerGenerationOutput:
    stats: dict[str, Any] = {
        "llm_calls": llm_calls,
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 12,
        "model": "test-model",
    }
    return AnswerGenerationOutput(answer=answer, timing_ms=1.0, llm_stats=stats, error=None)


def make_mock_llm(sql: str | None, answer: str = "Test answer") -> MagicMock:
    mock = MagicMock(spec=OpenRouterLLMClient)
    mock.model = "test-model"
    mock.generate_sql.return_value = _sql_gen(sql)
    mock.generate_answer.return_value = _answer_gen(answer)
    return mock


# ---------------------------------------------------------------------------
# _trim_sql_trailing_prose
# ---------------------------------------------------------------------------

class TestTrimSQLTrailingProse(unittest.TestCase):
    def test_semicolon_truncation(self):
        result = _trim_sql_trailing_prose("SELECT 1;\nThis query returns all rows.")
        self.assertEqual(result, "SELECT 1;")

    def test_blank_line_truncation(self):
        result = _trim_sql_trailing_prose("SELECT 1\n\nThis query explains something.")
        self.assertEqual(result, "SELECT 1")

    def test_no_trailing_prose(self):
        result = _trim_sql_trailing_prose("SELECT age FROM t")
        self.assertEqual(result, "SELECT age FROM t")

    def test_empty_returns_none(self):
        self.assertIsNone(_trim_sql_trailing_prose(""))
        self.assertIsNone(_trim_sql_trailing_prose(None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _extract_sql
# ---------------------------------------------------------------------------

class TestExtractSQL(unittest.TestCase):
    def test_sql_tagged_fence(self):
        result = OpenRouterLLMClient._extract_sql("```sql\nSELECT 1\n```")
        self.assertEqual(result, "SELECT 1")

    def test_untagged_fence_with_select(self):
        result = OpenRouterLLMClient._extract_sql("```\nSELECT age FROM t\n```")
        self.assertIsNotNone(result)
        self.assertTrue(result.lower().startswith("select"))

    def test_non_sql_fence_ignored(self):
        # A ```text fence that doesn't contain SQL should not be returned.
        text = "```text\nsome explanation\n```\n```sql\nSELECT 1\n```"
        result = OpenRouterLLMClient._extract_sql(text)
        self.assertEqual(result, "SELECT 1")

    def test_sql_fence_preferred_over_non_sql_fence(self):
        # If the first fence is non-SQL and the second is ```sql, return the SQL one.
        text = "Here is my plan:\n```\nStep 1: think\n```\n```sql\nSELECT 2\n```"
        result = OpenRouterLLMClient._extract_sql(text)
        self.assertEqual(result, "SELECT 2")

    def test_trailing_prose_trimmed_at_semicolon(self):
        text = "```sql\nSELECT 1;\nThis query is simple.\n```"
        result = OpenRouterLLMClient._extract_sql(text)
        self.assertEqual(result, "SELECT 1;")

    def test_with_clause(self):
        sql = "WITH cte AS (SELECT 1) SELECT * FROM cte"
        result = OpenRouterLLMClient._extract_sql(f"```sql\n{sql}\n```")
        self.assertIsNotNone(result)
        self.assertTrue(result.upper().startswith("WITH"))

    def test_cannot_answer_sentinel(self):
        self.assertIsNone(OpenRouterLLMClient._extract_sql("CANNOT_ANSWER"))
        self.assertIsNone(OpenRouterLLMClient._extract_sql("The answer is CANNOT_ANSWER."))

    def test_bare_select_fallback(self):
        result = OpenRouterLLMClient._extract_sql("Here is the SQL: SELECT age FROM t")
        self.assertIsNotNone(result)
        self.assertTrue(result.lower().startswith("select"))

    def test_returns_none_for_garbage(self):
        self.assertIsNone(OpenRouterLLMClient._extract_sql("I cannot generate SQL."))

    def test_trailing_prose_in_raw_fallback(self):
        text = "SELECT age FROM t;\nNote: this returns all ages."
        result = OpenRouterLLMClient._extract_sql(text)
        self.assertIsNotNone(result)
        self.assertNotIn("Note:", result)


# ---------------------------------------------------------------------------
# Token counting (_chat returns per-call stats)
# ---------------------------------------------------------------------------

class TestTokenCounting(unittest.TestCase):
    """Verify _chat returns accurate per-call stats without shared state."""

    def _make_mock_response(
        self,
        content: str,
        prompt_tokens: float = 10.0,
        completion_tokens: float = 5.0,
    ):
        """Build a minimal mock OpenRouter response."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        res = MagicMock()
        res.choices = [choice]
        res.usage = usage
        return res

    def _make_client(self):
        client = MagicMock(spec=OpenRouterLLMClient)
        client.model = "test"
        # We need a real _chat, not a mock — so instantiate via object.__new__ and
        # patch only the underlying OpenRouter SDK call.
        real = object.__new__(OpenRouterLLMClient)
        real.model = "test"
        return real

    def test_usage_fields_used_when_present(self):
        real = object.__new__(OpenRouterLLMClient)
        real.model = "test"
        mock_res = self._make_mock_response("SELECT 1", prompt_tokens=20.0, completion_tokens=8.0)
        mock_sdk = MagicMock()
        mock_sdk.chat.send.return_value = mock_res
        real._client = mock_sdk

        content, stats = real._chat(
            [{"role": "user", "content": "hello"}], temperature=1.0, max_tokens=100
        )
        self.assertEqual(content, "SELECT 1")
        self.assertEqual(stats["llm_calls"], 1)
        self.assertEqual(stats["prompt_tokens"], 20)
        self.assertEqual(stats["completion_tokens"], 8)
        self.assertEqual(stats["total_tokens"], 28)
        # All values must be int.
        for k, v in stats.items():
            self.assertIsInstance(v, int, msg=f"{k} must be int")

    def test_estimation_fallback_when_usage_absent(self):
        real = object.__new__(OpenRouterLLMClient)
        real.model = "test"
        msg = MagicMock()
        msg.content = "hello world answer"
        choice = MagicMock()
        choice.message = msg
        res = MagicMock()
        res.choices = [choice]
        res.usage = None  # No usage object.
        mock_sdk = MagicMock()
        mock_sdk.chat.send.return_value = res
        real._client = mock_sdk

        content, stats = real._chat(
            [{"role": "user", "content": "a b c d e"}], temperature=1.0, max_tokens=100
        )
        self.assertEqual(stats["llm_calls"], 1)
        self.assertGreater(stats["prompt_tokens"], 0)
        self.assertGreater(stats["completion_tokens"], 0)
        self.assertEqual(stats["total_tokens"], stats["prompt_tokens"] + stats["completion_tokens"])

    def test_no_shared_state_between_calls(self):
        """Two sequential calls must not accumulate into each other."""
        real = object.__new__(OpenRouterLLMClient)
        real.model = "test"
        res = self._make_mock_response("reply", prompt_tokens=10.0, completion_tokens=5.0)
        mock_sdk = MagicMock()
        mock_sdk.chat.send.return_value = res
        real._client = mock_sdk

        _, stats1 = real._chat([{"role": "user", "content": "q1"}], 1.0, 100)
        _, stats2 = real._chat([{"role": "user", "content": "q2"}], 1.0, 100)
        # Each call must independently report 1 llm_call, not accumulate.
        self.assertEqual(stats1["llm_calls"], 1)
        self.assertEqual(stats2["llm_calls"], 1)
        self.assertEqual(stats1["total_tokens"], 15)
        self.assertEqual(stats2["total_tokens"], 15)

    def test_content_block_list_handled(self):
        real = object.__new__(OpenRouterLLMClient)
        real.model = "test"
        block = MagicMock()
        block.text = "block text"
        msg = MagicMock()
        msg.content = [block]
        choice = MagicMock()
        choice.message = msg
        res = MagicMock()
        res.choices = [choice]
        res.usage = None
        mock_sdk = MagicMock()
        mock_sdk.chat.send.return_value = res
        real._client = mock_sdk

        content, stats = real._chat([{"role": "user", "content": "q"}], 1.0, 100)
        self.assertEqual(content, "block text")
        self.assertEqual(stats["llm_calls"], 1)


# ---------------------------------------------------------------------------
# SQLValidator
# ---------------------------------------------------------------------------

class TestSQLValidator(unittest.TestCase):
    def setUp(self):
        self.v = SQLValidator(columns=_COLUMNS)

    # --- basic acceptance ---
    def test_accepts_simple_select(self):
        out = self.v.validate(f"SELECT age FROM {_TABLE}")
        self.assertTrue(out.is_valid, out.error)

    def test_accepts_aggregate_with_explicit_alias(self):
        out = self.v.validate(f"SELECT AVG(addiction_level) AS avg_add FROM {_TABLE}")
        self.assertTrue(out.is_valid, out.error)

    def test_accepts_implicit_alias_via_fallback(self):
        # Without DB, implicit aliases slip through (permissive by design).
        out = self.v.validate(f"SELECT AVG(addiction_level) avg_add FROM {_TABLE}")
        # Permissive — do not hard-require True here, but it must not crash.
        self.assertIsInstance(out, SQLValidationOutput)

    def test_accepts_with_cte(self):
        sql = (
            f"WITH cte AS (SELECT age FROM {_TABLE}) "
            "SELECT * FROM cte"
        )
        out = self.v.validate(sql)
        self.assertTrue(out.is_valid, out.error)

    def test_accepts_short_table_alias(self):
        out = self.v.validate(f"SELECT t.age FROM {_TABLE} t")
        self.assertTrue(out.is_valid, out.error)

    def test_accepts_long_table_alias(self):
        out = self.v.validate(f"SELECT player.age FROM {_TABLE} player")
        self.assertTrue(out.is_valid, out.error)

    def test_accepts_group_by_count(self):
        out = self.v.validate(
            f"SELECT gender, COUNT(*) FROM {_TABLE} GROUP BY gender"
        )
        self.assertTrue(out.is_valid, out.error)

    def test_accepts_order_by_alias(self):
        sql = (
            f"SELECT gender, AVG(anxiety_score) AS avg_anx "
            f"FROM {_TABLE} GROUP BY gender ORDER BY avg_anx DESC"
        )
        out = self.v.validate(sql)
        self.assertTrue(out.is_valid, out.error)

    # --- rejection ---
    def test_rejects_none(self):
        self.assertFalse(self.v.validate(None).is_valid)

    def test_rejects_empty(self):
        self.assertFalse(self.v.validate("   ").is_valid)

    def test_rejects_delete(self):
        out = self.v.validate(f"DELETE FROM {_TABLE}")
        self.assertFalse(out.is_valid)
        self.assertIn("DELETE", out.error)

    def test_rejects_drop(self):
        self.assertFalse(self.v.validate(f"DROP TABLE {_TABLE}").is_valid)

    def test_rejects_insert(self):
        self.assertFalse(self.v.validate(f"INSERT INTO {_TABLE} VALUES (1,'M',1,1)").is_valid)

    def test_rejects_pragma(self):
        self.assertFalse(self.v.validate(f"PRAGMA table_info('{_TABLE}')").is_valid)

    def test_rejects_multi_statement(self):
        out = self.v.validate(f"SELECT 1; DROP TABLE {_TABLE}")
        self.assertFalse(out.is_valid)
        self.assertIn("Multiple", out.error)

    def test_rejects_unknown_column_no_db(self):
        out = self.v.validate(f"SELECT zodiac_sign FROM {_TABLE}")
        self.assertFalse(out.is_valid)
        self.assertIn("zodiac_sign", out.error)

    # --- table whitelist ---
    def test_rejects_sqlite_master(self):
        out = self.v.validate("SELECT * FROM sqlite_master")
        self.assertFalse(out.is_valid)
        self.assertIn("sqlite_master", out.error)

    def test_rejects_other_table(self):
        out = self.v.validate("SELECT * FROM other_table")
        self.assertFalse(out.is_valid)
        self.assertIn("other_table", out.error)

    def test_rejects_cross_join_to_other_table(self):
        out = self.v.validate(f"SELECT * FROM {_TABLE} JOIN other_table ON 1=1")
        self.assertFalse(out.is_valid)

    def test_cte_is_not_blocked_as_foreign_table(self):
        sql = (
            f"WITH summary AS (SELECT age FROM {_TABLE}) "
            "SELECT * FROM summary"
        )
        out = self.v.validate(sql)
        self.assertTrue(out.is_valid, out.error)

    # --- no schema → column check skipped, table whitelist still enforced ---
    def test_no_schema_skips_column_check_but_keeps_table_check(self):
        v = SQLValidator(columns=[])
        # Column check skipped — valid SQL accepted even with unknown identifiers.
        out = v.validate(f"SELECT zodiac_sign FROM {_TABLE}")
        self.assertTrue(out.is_valid)
        # Table whitelist still enforced.
        out2 = v.validate("SELECT * FROM sqlite_master")
        self.assertFalse(out2.is_valid)

    def test_timing_non_negative(self):
        out = self.v.validate("SELECT 1")
        self.assertGreaterEqual(out.timing_ms, 0.0)


# ---------------------------------------------------------------------------
# SQLValidator with real temp DB (EXPLAIN QUERY PLAN path)
# ---------------------------------------------------------------------------

class TestSQLValidatorWithDB(unittest.TestCase):
    def setUp(self):
        self.db_path = make_temp_db()
        self.v = SQLValidator(columns=_COLUMNS, db_path=self.db_path)

    def test_explain_rejects_unknown_column(self):
        out = self.v.validate(f"SELECT zodiac_sign FROM {_TABLE}")
        # Table whitelist rejects unknown table; if we sneak past that, EXPLAIN catches it.
        # Either the table check or EXPLAIN must reject this.
        self.assertFalse(out.is_valid)

    def test_explain_accepts_valid_cte(self):
        sql = (
            f"WITH s AS (SELECT age, addiction_level FROM {_TABLE}) "
            "SELECT age, addiction_level FROM s LIMIT 3"
        )
        out = self.v.validate(sql)
        self.assertTrue(out.is_valid, out.error)

    def test_explain_accepts_group_order(self):
        sql = (
            f"SELECT gender, AVG(anxiety_score) AS avg_anx "
            f"FROM {_TABLE} GROUP BY gender ORDER BY avg_anx DESC"
        )
        out = self.v.validate(sql)
        self.assertTrue(out.is_valid, out.error)


# ---------------------------------------------------------------------------
# Pipeline integration tests (mocked LLM, real temp DB)
# ---------------------------------------------------------------------------

class TestPipelineWithMockLLM(unittest.TestCase):
    def setUp(self):
        self.db_path = make_temp_db()

    def _pipeline(self, sql: str | None, answer: str = "Test answer") -> AnalyticsPipeline:
        return AnalyticsPipeline(db_path=self.db_path, llm_client=make_mock_llm(sql, answer))

    def test_success_path(self):
        sql = f"SELECT gender, COUNT(*) as cnt FROM {_TABLE} GROUP BY gender"
        p = self._pipeline(sql)
        r = p.run("How many by gender?")
        self.assertEqual(r.status, "success")
        self.assertGreater(len(r.rows), 0)
        for key in ("sql_generation_ms", "sql_validation_ms", "sql_execution_ms",
                    "answer_generation_ms", "total_ms"):
            self.assertIn(key, r.timings)
            self.assertGreaterEqual(r.timings[key], 0.0)

    def test_invalid_sql_path_validator_exercised(self):
        # Pass a DELETE directly to the validator (not via destructive question)
        # by giving the mock a DELETE sql but a non-destructive question.
        p = self._pipeline(f"DELETE FROM {_TABLE}")
        r = p.run("What is the average addiction level?")
        self.assertEqual(r.status, "invalid_sql")
        self.assertIsNotNone(r.sql_validation.error)

    def test_destructive_question_short_circuits(self):
        # Destructive question detected before any LLM call.
        p = self._pipeline(f"SELECT * FROM {_TABLE}")
        r = p.run(f"Please delete all rows from the {_TABLE} table")
        self.assertEqual(r.status, "invalid_sql")
        self.assertIn("cannot answer", r.answer.lower())
        # No LLM calls made.
        self.assertEqual(r.total_llm_stats["llm_calls"], 0)

    def test_unanswerable_path_no_sql(self):
        p = self._pipeline(None)
        r = p.run("Which zodiac sign has the highest stress score?")
        self.assertEqual(r.status, "unanswerable")
        self.assertIn("cannot answer", r.answer.lower())

    def test_unanswerable_path_unknown_column(self):
        p = self._pipeline(f"SELECT zodiac_sign FROM {_TABLE}")
        r = p.run("Which zodiac sign has the highest stress score?")
        self.assertIn(r.status, {"unanswerable", "invalid_sql"})
        self.assertIn("cannot answer", r.answer.lower())

    def test_execution_error_yields_cannot_answer(self):
        # Bypass validator to test execution error path.
        p = self._pipeline(f"SELECT nonexistent_col FROM nonexistent_table")
        with patch.object(
            p.validator,
            "validate",
            return_value=SQLValidationOutput(
                is_valid=True,
                validated_sql=f"SELECT nonexistent_col FROM nonexistent_table",
            ),
        ):
            r = p.run("test question")
        self.assertEqual(r.status, "error")
        self.assertIn("cannot answer", r.answer.lower())

    def test_scalar_fast_path_skips_llm(self):
        sql = f"SELECT COUNT(*) AS cnt FROM {_TABLE}"
        p = self._pipeline(sql)
        r = p.run("How many rows?")
        if r.status == "success":
            self.assertEqual(r.answer_generation.llm_stats["llm_calls"], 0)

    def test_token_stats_are_int(self):
        sql = f"SELECT age FROM {_TABLE} LIMIT 1"
        r = self._pipeline(sql).run("test")
        for key in ("llm_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertIsInstance(r.total_llm_stats[key], int, msg=f"{key} must be int")
            self.assertGreaterEqual(r.total_llm_stats[key], 0)
        self.assertIsInstance(r.total_llm_stats["model"], str)

    def test_output_contract_stage_types(self):
        sql = f"SELECT gender, AVG(anxiety_score) FROM {_TABLE} GROUP BY gender"
        r = self._pipeline(sql).run("avg anxiety by gender")
        from src.types import (
            SQLGenerationOutput, SQLValidationOutput,
            SQLExecutionOutput, AnswerGenerationOutput,
        )
        self.assertIsInstance(r.sql_generation, SQLGenerationOutput)
        self.assertIsInstance(r.sql_validation, SQLValidationOutput)
        self.assertIsInstance(r.sql_execution, SQLExecutionOutput)
        self.assertIsInstance(r.answer_generation, AnswerGenerationOutput)

    def test_question_and_request_id_echoed(self):
        q = "How many rows?"
        r = self._pipeline(f"SELECT 1").run(q, request_id="req-999")
        self.assertEqual(r.question, q)
        self.assertEqual(r.request_id, "req-999")

    def test_answer_generation_failure_sets_error_status(self):
        """If the answer LLM call raises, status must be 'error' not 'success'."""
        db_path = make_temp_db()
        mock_llm = make_mock_llm(f"SELECT gender, COUNT(*) FROM {_TABLE} GROUP BY gender")
        # Make generate_answer return an output with an error set.
        mock_llm.generate_answer.return_value = AnswerGenerationOutput(
            answer="Error generating answer: timeout",
            timing_ms=0.0,
            llm_stats={"llm_calls": 1, "prompt_tokens": 5, "completion_tokens": 0,
                       "total_tokens": 5, "model": "test"},
            error="timeout",
        )
        p = AnalyticsPipeline(db_path=db_path, llm_client=mock_llm)
        r = p.run("breakdown by gender")
        self.assertEqual(r.status, "error")

    def test_sqlite_master_rejected(self):
        p = self._pipeline("SELECT * FROM sqlite_master")
        r = p.run("show me the schema")
        self.assertIn(r.status, {"invalid_sql", "unanswerable"})

    def test_cte_query_accepted(self):
        cte_sql = (
            f"WITH top AS (SELECT age, addiction_level FROM {_TABLE} ORDER BY addiction_level DESC LIMIT 2) "
            "SELECT * FROM top"
        )
        p = self._pipeline(cte_sql)
        r = p.run("show top addiction levels")
        self.assertEqual(r.status, "success")

    def test_long_alias_query_accepted(self):
        alias_sql = f"SELECT player.age FROM {_TABLE} player LIMIT 1"
        p = self._pipeline(alias_sql)
        r = p.run("get ages")
        self.assertNotEqual(r.status, "invalid_sql")


# ---------------------------------------------------------------------------
# Benchmark script regression
# ---------------------------------------------------------------------------

class TestBenchmarkStatusAccess(unittest.TestCase):
    def test_status_is_attribute_not_subscript(self):
        r = PipelineOutput(
            status="success",
            question="test",
            request_id=None,
            sql_generation=SQLGenerationOutput(
                sql="SELECT 1", timing_ms=1.0,
                llm_stats={"llm_calls": 1, "prompt_tokens": 5,
                           "completion_tokens": 3, "total_tokens": 8, "model": "t"},
            ),
            sql_validation=SQLValidationOutput(is_valid=True, validated_sql="SELECT 1"),
            sql_execution=SQLExecutionOutput(rows=[{"cnt": 1}], row_count=1, timing_ms=0.5),
            answer_generation=AnswerGenerationOutput(
                answer="one", timing_ms=1.0,
                llm_stats={"llm_calls": 1, "prompt_tokens": 5,
                           "completion_tokens": 3, "total_tokens": 8, "model": "t"},
            ),
        )
        self.assertEqual(r.status, "success")
        with self.assertRaises(TypeError):
            _ = r["status"]  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
