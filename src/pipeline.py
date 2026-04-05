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

# The only table this pipeline is authorised to query.
_ALLOWED_TABLE = "gaming_mental_health"

# Phrases in the user question that indicate destructive intent.
# The LLM follows the system prompt and generates SELECT queries even for these,
# so we must check the original question directly.
_DESTRUCTIVE_QUESTION_PATTERNS = re.compile(
    r"\b(delete|drop|truncate|remove all|erase|wipe|clear all|destroy)\b",
    re.IGNORECASE,
)

# SQL statement types that are never allowed regardless of context.
_BLOCKED_KEYWORDS = frozenset({
    "DELETE", "INSERT", "UPDATE", "DROP", "CREATE", "ALTER", "TRUNCATE",
    "REPLACE", "PRAGMA", "ATTACH", "DETACH", "VACUUM",
})

# SQL reserved words that must not be treated as column names during validation.
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
    "OVER", "PARTITION", "FILTER", "NULLS", "FIRST", "LAST", "TIES",
    "FETCH", "NEXT", "ROWS", "ONLY", "WINDOW", "RANGE",
})


class SQLValidationError(Exception):
    pass


class SQLValidator:
    """Schema-aware SQL validator. Instantiate once per pipeline with known schema."""

    def __init__(
        self,
        columns: list[str] | None = None,
        db_path: Path | None = None,
        table_name: str = _ALLOWED_TABLE,
    ) -> None:
        self._columns: frozenset[str] = frozenset(c.lower() for c in (columns or []))
        self._db_path = db_path
        self._table_name = table_name.lower()
        # Table name word-parts are not column names.
        self._table_tokens: frozenset[str] = frozenset(
            self._table_name.split("_") + [self._table_name]
        )

    def validate(self, sql: str | None) -> SQLValidationOutput:
        start = time.perf_counter()

        if not sql or not sql.strip():
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error="No SQL provided",
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        # Strip comments for static analysis (original sql is preserved for execution).
        clean = re.sub(r"--[^\n]*", " ", sql)
        clean = re.sub(r"/\*.*?\*/", " ", clean, flags=re.DOTALL)
        clean = clean.strip()

        # Reject multiple statements.
        parts = [p.strip() for p in clean.split(";") if p.strip()]
        if len(parts) > 1:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error="Multiple SQL statements are not allowed",
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        # Reject blocked statement types.
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

        # Table whitelist: only the allowed table may be referenced.
        table_error = self._check_tables(clean)
        if table_error:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=table_error,
                timing_ms=(time.perf_counter() - start) * 1000,
            )

        # EXPLAIN QUERY PLAN is the primary structural validator when the DB is present.
        # It catches wrong columns, type errors, and disallowed functions (e.g. load_extension).
        if self._db_path and self._db_path.exists():
            explain_error = self._explain_check(sql)
            if explain_error:
                return SQLValidationOutput(
                    is_valid=False,
                    validated_sql=None,
                    error=explain_error,
                    timing_ms=(time.perf_counter() - start) * 1000,
                )
        elif self._columns:
            # DB unavailable — fall back to regex column check as a best-effort filter.
            col_error = self._check_columns(clean)
            if col_error:
                return SQLValidationOutput(
                    is_valid=False,
                    validated_sql=None,
                    error=col_error,
                    timing_ms=(time.perf_counter() - start) * 1000,
                )

        return SQLValidationOutput(
            is_valid=True,
            validated_sql=sql,
            error=None,
            timing_ms=(time.perf_counter() - start) * 1000,
        )

    def _check_tables(self, clean_sql: str) -> str | None:
        """Reject any query that references a table other than the allowed one.

        CTE pseudo-table names are whitelisted automatically so that
        `WITH cte AS (...) SELECT * FROM cte` is accepted.
        """
        # Collect CTE names — they are not real tables.
        cte_names: frozenset[str] = frozenset(
            m.lower()
            for m in re.findall(
                r"\bWITH\s+([a-z_][a-z0-9_]*)\s+AS\s*\(",
                clean_sql,
                flags=re.IGNORECASE,
            )
        )

        # Find all identifiers that follow FROM or JOIN.
        table_refs = re.findall(
            r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)\b",
            clean_sql,
            flags=re.IGNORECASE,
        )

        for ref in table_refs:
            ref_lower = ref.lower()
            if ref_lower in cte_names:
                continue
            if ref_lower == self._table_name:
                continue
            return (
                f"Query references disallowed table or object '{ref}'. "
                f"Only '{self._table_name}' is permitted."
            )
        return None

    def _check_columns(self, clean_sql: str) -> str | None:
        """Best-effort column reference check used only when the DB is unavailable.

        Handles:
        - Explicit aliases (AS name) — collected and whitelisted.
        - CTE names — collected and whitelisted.
        - Table aliases — collected from FROM/JOIN clauses and whitelisted.
        - Qualified references (alias.column) — qualifier prefix stripped, only column checked.
        - Implicit aliases (expr name without AS) — not detected; check is permissive.
        """
        # Collect explicit aliases.
        explicit_aliases: frozenset[str] = frozenset(
            m.lower()
            for m in re.findall(
                r"\bAS\s+([a-z_][a-z0-9_]*)\b", clean_sql, flags=re.IGNORECASE
            )
        )

        # Collect CTE names.
        cte_names: frozenset[str] = frozenset(
            m.lower()
            for m in re.findall(
                r"\bWITH\s+([a-z_][a-z0-9_]*)\s+AS\s*\(",
                clean_sql,
                flags=re.IGNORECASE,
            )
        )

        # Collect table aliases from FROM/JOIN table alias patterns.
        # Only treat the token as an alias if it's not a SQL keyword.
        raw_aliases = re.findall(
            r"\b(?:FROM|JOIN)\s+[a-z_][a-z0-9_]*\s+([a-z_][a-z0-9_]*)\b",
            clean_sql,
            flags=re.IGNORECASE,
        )
        table_aliases: frozenset[str] = frozenset(
            a.lower() for a in raw_aliases if a.upper() not in _SQL_RESERVED
        )

        known = (
            self._columns
            | explicit_aliases
            | cte_names
            | table_aliases
            | self._table_tokens
        )

        # Strip string literals and quoted identifiers.
        no_strings = re.sub(r"'[^']*'", " ", clean_sql)
        no_quoted = re.sub(r'"[^"]*"', " ", no_strings)
        no_backtick = re.sub(r"`[^`]*`", " ", no_quoted)
        # Remove AS alias declarations at their definition site.
        no_alias_decls = re.sub(
            r"\bAS\s+[a-z_][a-z0-9_]*\b", " ", no_backtick, flags=re.IGNORECASE
        )
        # For qualified refs (alias.column), strip the qualifier prefix so only
        # the column name is checked, not the table alias.
        no_qualifiers = re.sub(r"\b[a-z_][a-z0-9_]*\.", "", no_alias_decls, flags=re.IGNORECASE)

        identifiers = re.findall(r"\b([a-z][a-z0-9_]*)\b", no_qualifiers.lower())

        unknown: list[str] = []
        for ident in set(identifiers):
            if ident in _SQL_RESERVED or ident.upper() in _SQL_RESERVED:
                continue
            if len(ident) <= 2:
                continue
            if ident in known:
                continue
            unknown.append(ident)

        if unknown:
            return (
                f"Query references column(s) not in schema: {', '.join(sorted(unknown))}. "
                "Please rephrase using known survey fields."
            )
        return None

    def _explain_check(self, sql: str) -> str | None:
        """Run EXPLAIN QUERY PLAN to catch structural errors SQLite would reject."""
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

        error: str | None = None
        rows: list[dict] = []
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
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        llm_client: OpenRouterLLMClient | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.llm = llm_client or build_default_llm_client()
        self.executor = SQLiteExecutor(self.db_path)

        # Schema loaded once at construction; shared between validator and SQL prompt.
        self._schema = self._load_schema()
        table_name = self._schema.get("table", _ALLOWED_TABLE)
        columns = [c["name"] for c in self._schema.get("columns", [])]
        self.validator = SQLValidator(
            columns=columns, db_path=self.db_path, table_name=table_name
        )

    def _load_schema(self) -> dict:
        """Introspect the SQLite DB and return table/column metadata."""
        table = _ALLOWED_TABLE
        if not self.db_path.exists():
            logger.warning(
                json.dumps({
                    "event": "schema_load_skipped",
                    "reason": "db_not_found",
                    "db_path": str(self.db_path),
                })
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
                json.dumps({
                    "event": "schema_loaded",
                    "table": table,
                    "column_count": len(columns),
                })
            )
            return {"table": table, "columns": columns}
        except Exception as exc:
            logger.warning(json.dumps({"event": "schema_load_error", "error": str(exc)}))
            return {}

    def run(self, question: str, request_id: str | None = None) -> PipelineOutput:
        start = time.perf_counter()
        _llm_model = getattr(self.llm, "model", "unknown")
        _zero_stats: dict = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": _llm_model,
        }

        logger.info(
            json.dumps({
                "event": "pipeline_start",
                "request_id": request_id,
                "question": question[:80],
                "model": _llm_model,
            })
        )

        # Pre-check: reject questions with clear destructive intent before any LLM call.
        if _DESTRUCTIVE_QUESTION_PATTERNS.search(question):
            _t = (time.perf_counter() - start) * 1000
            _gen = SQLGenerationOutput(
                sql=None,
                timing_ms=_t,
                llm_stats=dict(_zero_stats),
                error="Destructive intent detected in question",
            )
            _val = SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error="Question contains destructive intent — only read queries are supported",
                timing_ms=0.0,
            )
            _exe = SQLExecutionOutput(rows=[], row_count=0, timing_ms=0.0, error=None)
            _ans = AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=dict(_zero_stats),
                error=None,
            )
            _timings = {
                "sql_generation_ms": _t,
                "sql_validation_ms": 0.0,
                "sql_execution_ms": 0.0,
                "answer_generation_ms": 0.0,
                "total_ms": (time.perf_counter() - start) * 1000,
            }
            logger.info(
                json.dumps({
                    "event": "pipeline_complete",
                    "request_id": request_id,
                    "status": "invalid_sql",
                    "reason": "destructive_intent",
                    "total_ms": round(_timings["total_ms"], 1),
                })
            )
            return PipelineOutput(
                status="invalid_sql",
                question=question,
                request_id=request_id,
                sql_generation=_gen,
                sql_validation=_val,
                sql_execution=_exe,
                answer_generation=_ans,
                sql=None,
                rows=[],
                answer=_ans.answer,
                timings=_timings,
                total_llm_stats=dict(_zero_stats),
            )

        # Stage 1: SQL Generation
        logger.debug(json.dumps({"event": "stage_start", "stage": "sql_generation", "request_id": request_id}))
        sql_gen_output = self.llm.generate_sql(question, self._schema)
        sql = sql_gen_output.sql
        logger.debug(
            json.dumps({
                "event": "stage_complete",
                "stage": "sql_generation",
                "request_id": request_id,
                "sql_preview": (sql[:120] if sql else None),
                "llm_calls": sql_gen_output.llm_stats.get("llm_calls", 0),
                "total_tokens": sql_gen_output.llm_stats.get("total_tokens", 0),
                "error": sql_gen_output.error,
                "timing_ms": round(sql_gen_output.timing_ms, 1),
            })
        )

        # Stage 2: SQL Validation
        logger.debug(json.dumps({"event": "stage_start", "stage": "sql_validation", "request_id": request_id}))
        validation_output = self.validator.validate(sql)
        if not validation_output.is_valid:
            sql = None
        logger.debug(
            json.dumps({
                "event": "stage_complete",
                "stage": "sql_validation",
                "request_id": request_id,
                "is_valid": validation_output.is_valid,
                "error": validation_output.error,
                "timing_ms": round(validation_output.timing_ms, 1),
            })
        )

        # Stage 3: SQL Execution
        logger.debug(json.dumps({"event": "stage_start", "stage": "sql_execution", "request_id": request_id}))
        execution_output = self.executor.run(sql)
        logger.debug(
            json.dumps({
                "event": "stage_complete",
                "stage": "sql_execution",
                "request_id": request_id,
                "row_count": execution_output.row_count,
                "error": execution_output.error,
                "timing_ms": round(execution_output.timing_ms, 1),
            })
        )

        # Stage 4: Answer Generation.
        # All routing decisions live here (not in the LLM client) so mocked and real
        # clients behave identically.
        logger.debug(json.dumps({"event": "stage_start", "stage": "answer_generation", "request_id": request_id}))
        if sql is None:
            answer_output = AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=dict(_zero_stats),
                error=None,
            )
            answer_fast_path = "no_sql"
        elif execution_output.error:
            answer_output = AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=dict(_zero_stats),
                error=execution_output.error,
            )
            answer_fast_path = "execution_error"
        elif len(execution_output.rows) == 1 and len(execution_output.rows[0]) == 1:
            row = execution_output.rows[0]
            value = next(iter(row.values()))
            col = next(iter(row.keys()))
            answer_output = AnswerGenerationOutput(
                answer=f"The result for '{col}' is: {value}",
                timing_ms=0.0,
                llm_stats=dict(_zero_stats),
                error=None,
            )
            answer_fast_path = "scalar"
        else:
            answer_output = self.llm.generate_answer(question, sql, execution_output.rows)
            answer_fast_path = "llm"

        logger.debug(
            json.dumps({
                "event": "stage_complete",
                "stage": "answer_generation",
                "request_id": request_id,
                "fast_path": answer_fast_path,
                "llm_calls": answer_output.llm_stats.get("llm_calls", 0),
                "error": answer_output.error,
                "timing_ms": round(answer_output.timing_ms, 1),
            })
        )

        # Determine status — explicit priority chain.
        if sql_gen_output.sql is None:
            status = "unanswerable"
        elif not validation_output.is_valid:
            status = "invalid_sql"
        elif execution_output.error:
            status = "error"
        elif answer_output.error:
            # Answer generation itself failed (e.g. API error during answer call).
            status = "error"
        else:
            status = "success"

        timings = {
            "sql_generation_ms": sql_gen_output.timing_ms,
            "sql_validation_ms": validation_output.timing_ms,
            "sql_execution_ms": execution_output.timing_ms,
            "answer_generation_ms": answer_output.timing_ms,
            "total_ms": (time.perf_counter() - start) * 1000,
        }

        total_llm_stats = {
            "llm_calls": int(
                sql_gen_output.llm_stats.get("llm_calls", 0)
                + answer_output.llm_stats.get("llm_calls", 0)
            ),
            "prompt_tokens": int(
                sql_gen_output.llm_stats.get("prompt_tokens", 0)
                + answer_output.llm_stats.get("prompt_tokens", 0)
            ),
            "completion_tokens": int(
                sql_gen_output.llm_stats.get("completion_tokens", 0)
                + answer_output.llm_stats.get("completion_tokens", 0)
            ),
            "total_tokens": int(
                sql_gen_output.llm_stats.get("total_tokens", 0)
                + answer_output.llm_stats.get("total_tokens", 0)
            ),
            "model": sql_gen_output.llm_stats.get("model", "unknown"),
        }

        logger.info(
            json.dumps({
                "event": "pipeline_complete",
                "request_id": request_id,
                "status": status,
                "total_ms": round(timings["total_ms"], 1),
                "total_tokens": total_llm_stats["total_tokens"],
                "llm_calls": total_llm_stats["llm_calls"],
                "row_count": execution_output.row_count,
                "answer_fast_path": answer_fast_path,
            })
        )

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
