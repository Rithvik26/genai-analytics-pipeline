from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path

from src.llm_client import OpenRouterLLMClient, build_default_llm_client
from src.types import (
    AnswerGenerationOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
    SQLExecutionOutput,
    PipelineOutput,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "data" / "gaming_mental_health.sqlite"

# Phrases in the user question that indicate destructive intent.
# The LLM follows the system prompt and generates SELECT queries even for these,
# so we must check the original question directly.
_DESTRUCTIVE_QUESTION_PATTERNS = re.compile(
    r"\b(delete|drop|truncate|remove all|erase|wipe|clear all|destroy)\b",
    re.IGNORECASE,
)

# SQL keywords that must never appear as the leading statement keyword.
_BLOCKED_KEYWORDS = frozenset({
    "DELETE", "INSERT", "UPDATE", "DROP", "CREATE", "ALTER", "TRUNCATE",
    "REPLACE", "PRAGMA", "ATTACH", "DETACH", "VACUUM",
})

# SQL reserved words that are NOT column names and should be excluded from
# column-reference checks.
_SQL_RESERVED = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "AS", "ON", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "DISTINCT",
    "UNION", "INTERSECT", "EXCEPT", "ALL", "CASE", "WHEN", "THEN", "ELSE",
    "END", "WITH", "RECURSIVE", "EXISTS", "BETWEEN", "LIKE", "GLOB",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE", "IFNULL", "CAST",
    "INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC", "ASC", "DESC",
    "ROWID", "OID", "TRUE", "FALSE", "PRIMARY", "KEY", "FOREIGN",
    "REFERENCES", "UNIQUE", "CHECK", "DEFAULT", "INDEX", "TABLE", "VIEW",
})


class SQLValidationError(Exception):
    pass


class SQLValidator:
    """Schema-aware SQL validator. Instantiate once with known columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        db_path: Path | None = None,
        table_name: str = "gaming_mental_health",
    ) -> None:
        self._columns: frozenset[str] = frozenset(c.lower() for c in (columns or []))
        self._db_path = db_path
        # Table name parts must not be treated as unknown column identifiers.
        self._table_tokens: frozenset[str] = frozenset(table_name.lower().split("_") + [table_name.lower()])

    def validate(self, sql: str | None) -> SQLValidationOutput:
        start = time.perf_counter()

        if not sql or not sql.strip():
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error="No SQL provided",
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        # Strip line comments and block comments for analysis only.
        clean = re.sub(r"--[^\n]*", " ", sql)
        clean = re.sub(r"/\*.*?\*/", " ", clean, flags=re.DOTALL)
        clean = clean.strip()

        # Reject multiple statements (semicolon followed by non-whitespace).
        parts = [p.strip() for p in clean.split(";") if p.strip()]
        if len(parts) > 1:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error="Multiple SQL statements are not allowed",
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        # Get leading keyword.
        first_token = clean.split()[0].upper() if clean.split() else ""
        if first_token in _BLOCKED_KEYWORDS:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=f"Statement type '{first_token}' is not allowed — only SELECT/WITH queries are permitted",
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        if first_token not in ("SELECT", "WITH"):
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=f"Only SELECT or WITH queries are allowed (got '{first_token}')",
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        # Column reference check (only when we have schema info).
        if self._columns:
            col_error = self._check_columns(clean)
            if col_error:
                return SQLValidationOutput(
                    is_valid=False,
                    validated_sql=None,
                    error=col_error,
                    timing_ms=(time.perf_counter() - start) * 1000,
                )

        # Secondary structural check via EXPLAIN QUERY PLAN when DB is available.
        if self._db_path and self._db_path.exists():
            explain_error = self._explain_check(sql)
            if explain_error:
                return SQLValidationOutput(
                    is_valid=False,
                    validated_sql=None,
                    error=explain_error,
                    timing_ms=(time.perf_counter() - start) * 1000,
                )

        return SQLValidationOutput(
            is_valid=True,
            validated_sql=sql,
            error=None,
            timing_ms=(time.perf_counter() - start) * 1000,
        )

    def _check_columns(self, clean_sql: str) -> str | None:
        """Return an error string if unknown column names are referenced, else None.

        Strips quoted identifiers and string literals first to avoid false positives
        from aliases, function names in strings, or literal values.
        """
        # Collect all alias names (after AS) first — they are valid references throughout
        # the query (e.g. ORDER BY uses aliases defined in SELECT).
        defined_aliases: frozenset[str] = frozenset(
            m.lower()
            for m in re.findall(r"\bAS\s+([a-z_][a-z0-9_]*)\b", clean_sql, flags=re.IGNORECASE)
        )

        # Remove single-quoted string literals.
        no_strings = re.sub(r"'[^']*'", " ", clean_sql)
        # Remove double-quoted identifiers (explicitly named; trust them).
        no_quoted = re.sub(r'"[^"]*"', " ", no_strings)
        # Remove backtick-quoted identifiers.
        no_backtick = re.sub(r"`[^`]*`", " ", no_quoted)
        # Remove AS <alias> declarations so the alias name isn't flagged at definition site.
        no_alias_decls = re.sub(r"\bAS\s+[a-z_][a-z0-9_]*\b", " ", no_backtick, flags=re.IGNORECASE)

        # Extract bare lowercase identifiers (snake_case style — likely column/alias refs).
        identifiers = re.findall(r"\b([a-z][a-z0-9_]*)\b", no_alias_decls.lower())

        unknown = []
        for ident in set(identifiers):
            if ident in _SQL_RESERVED or ident.upper() in _SQL_RESERVED:
                continue
            # Skip short generic tokens that are likely SQL syntax or aliases.
            if len(ident) <= 2:
                continue
            # Skip table name and its component words.
            if ident in self._table_tokens:
                continue
            # Skip aliases defined within this query (e.g. ORDER BY uses SELECT aliases).
            if ident in defined_aliases:
                continue
            if ident not in self._columns:
                unknown.append(ident)

        if unknown:
            return (
                f"Query references column(s) not in schema: {', '.join(sorted(unknown))}. "
                "Please rephrase using known survey fields."
            )
        return None

    def _explain_check(self, sql: str) -> str | None:
        """Run EXPLAIN QUERY PLAN to catch structural errors the parser missed."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            return None
        except sqlite3.OperationalError as exc:
            return f"SQL structural error: {exc}"
        except Exception as exc:
            return f"Validation error: {exc}"


class SQLiteExecutor:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def run(self, sql: str | None) -> SQLExecutionOutput:
        start = time.perf_counter()

        if sql is None:
            return SQLExecutionOutput(
                rows=[],
                row_count=0,
                timing_ms=(time.perf_counter() - start) * 1000,
                error=None,
            )

        error = None
        rows = []
        row_count = 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchmany(100)]
                row_count = len(rows)
        except Exception as exc:
            error = str(exc)
            rows = []
            row_count = 0

        return SQLExecutionOutput(
            rows=rows,
            row_count=row_count,
            timing_ms=(time.perf_counter() - start) * 1000,
            error=error,
        )


class AnalyticsPipeline:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, llm_client: OpenRouterLLMClient | None = None) -> None:
        self.db_path = Path(db_path)
        self.llm = llm_client or build_default_llm_client()
        self.executor = SQLiteExecutor(self.db_path)

        # Load schema once at construction time; used by both validator and SQL generation.
        self._schema = self._load_schema()

        table_name = self._schema.get("table", "gaming_mental_health")
        columns = [c["name"] for c in self._schema.get("columns", [])]
        self.validator = SQLValidator(columns=columns, db_path=self.db_path, table_name=table_name)

    def _load_schema(self) -> dict:
        """Introspect the SQLite DB and return table/column metadata."""
        table = "gaming_mental_health"
        if not self.db_path.exists():
            logger.warning(
                json.dumps({"event": "schema_load_skipped", "reason": "db_not_found", "db_path": str(self.db_path)})
            )
            return {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(f'PRAGMA table_info("{table}")')
                rows = cur.fetchall()
            if not rows:
                return {}
            columns = [{"name": r[1], "type": r[2]} for r in rows]
            logger.debug(
                json.dumps({"event": "schema_loaded", "table": table, "column_count": len(columns)})
            )
            return {"table": table, "columns": columns}
        except Exception as exc:
            logger.warning(json.dumps({"event": "schema_load_error", "error": str(exc)}))
            return {}

    def run(self, question: str, request_id: str | None = None) -> PipelineOutput:
        start = time.perf_counter()
        logger.info(json.dumps({"event": "pipeline_start", "request_id": request_id, "question": question[:80]}))

        # Pre-check: reject questions with clear destructive intent before any LLM call.
        # The LLM follows the system prompt and generates SELECT queries even for "delete all
        # rows" prompts, so we must check the original question directly.
        if _DESTRUCTIVE_QUESTION_PATTERNS.search(question):
            _t = (time.perf_counter() - start) * 1000
            _zero = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": getattr(self.llm, "model", "unknown")}
            _gen = SQLGenerationOutput(sql=None, timing_ms=_t, llm_stats=_zero, error="Destructive intent detected in question")
            _val = SQLValidationOutput(is_valid=False, validated_sql=None, error="Question contains destructive intent — only read queries are supported", timing_ms=0.0)
            _exe = SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error=None)
            _ans = AnswerGenerationOutput(answer="I cannot answer this question with the available data.", timing_ms=0.0, llm_stats=_zero, error=None)
            _timings = {"sql_generation_ms": _t, "sql_validation_ms": 0.0, "sql_execution_ms": 0.0, "answer_generation_ms": 0.0, "total_ms": (time.perf_counter() - start) * 1000}
            return PipelineOutput(status="invalid_sql", question=question, request_id=request_id, sql_generation=_gen, sql_validation=_val, sql_execution=_exe, answer_generation=_ans, sql=None, rows=[], answer=_ans.answer, timings=_timings, total_llm_stats={**_zero})

        # Stage 1: SQL Generation
        sql_gen_output = self.llm.generate_sql(question, self._schema)
        sql = sql_gen_output.sql

        # Stage 2: SQL Validation
        validation_output = self.validator.validate(sql)
        if not validation_output.is_valid:
            sql = None

        # Stage 3: SQL Execution
        execution_output = self.executor.run(sql)

        # Stage 4: Answer Generation.
        # Short-circuit in the pipeline (not in the LLM client) so mocked LLMs and
        # real LLMs behave identically for unanswerable / error paths.
        _llm_model = getattr(self.llm, "model", "unknown")
        _zero_stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": _llm_model}
        if sql is None:
            answer_output = AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=_zero_stats,
                error=None,
            )
        elif execution_output.error:
            answer_output = AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=_zero_stats,
                error=execution_output.error,
            )
        elif len(execution_output.rows) == 1 and len(execution_output.rows[0]) == 1:
            # Scalar fast-path: single value — no LLM call needed.
            row = execution_output.rows[0]
            value = next(iter(row.values()))
            col = next(iter(row.keys()))
            answer_output = AnswerGenerationOutput(
                answer=f"The result for '{col}' is: {value}",
                timing_ms=0.0,
                llm_stats=_zero_stats,
                error=None,
            )
        else:
            answer_output = self.llm.generate_answer(
                question,
                sql,
                execution_output.rows,
            )

        # Determine status — explicit priority chain.
        if sql_gen_output.sql is None:
            status = "unanswerable"
        elif not validation_output.is_valid:
            status = "invalid_sql"
        elif execution_output.error:
            status = "error"
        else:
            status = "success"

        # Build timings aggregate.
        timings = {
            "sql_generation_ms": sql_gen_output.timing_ms,
            "sql_validation_ms": validation_output.timing_ms,
            "sql_execution_ms": execution_output.timing_ms,
            "answer_generation_ms": answer_output.timing_ms,
            "total_ms": (time.perf_counter() - start) * 1000,
        }

        # Build total LLM stats — ensure all values are int.
        total_llm_stats = {
            "llm_calls": int(sql_gen_output.llm_stats.get("llm_calls", 0) + answer_output.llm_stats.get("llm_calls", 0)),
            "prompt_tokens": int(sql_gen_output.llm_stats.get("prompt_tokens", 0) + answer_output.llm_stats.get("prompt_tokens", 0)),
            "completion_tokens": int(sql_gen_output.llm_stats.get("completion_tokens", 0) + answer_output.llm_stats.get("completion_tokens", 0)),
            "total_tokens": int(sql_gen_output.llm_stats.get("total_tokens", 0) + answer_output.llm_stats.get("total_tokens", 0)),
            "model": sql_gen_output.llm_stats.get("model", "unknown"),
        }

        logger.info(json.dumps({
            "event": "pipeline_complete",
            "request_id": request_id,
            "status": status,
            "total_ms": round(timings["total_ms"], 1),
            "total_tokens": total_llm_stats["total_tokens"],
        }))

        return PipelineOutput(
            status=status,
            question=question,
            request_id=request_id,
            sql_generation=sql_gen_output,
            sql_validation=validation_output,
            sql_execution=execution_output,
            answer_generation=answer_output,
            sql=sql,
            rows=execution_output.rows,
            answer=answer_output.answer,
            timings=timings,
            total_llm_stats=total_llm_stats,
        )
