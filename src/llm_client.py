from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from src.types import SQLGenerationOutput, AnswerGenerationOutput

DEFAULT_MODEL = "openai/gpt-5-nano"

# SQL keywords used for trailing-prose detection in _trim_sql_trailing_prose.
_SQL_LINE_STARTERS = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL",
    "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "UNION",
    "WITH", "CASE", "WHEN", "THEN", "ELSE", "END", "ON",
    "INTERSECT", "EXCEPT", "--", "/*",
})


def _trim_sql_trailing_prose(text: str) -> str | None:
    """Trim trailing natural-language prose from an extracted SQL fragment.

    Strategy:
    - If a semicolon is present, truncate at the first semicolon (end of statement).
    - Otherwise, drop lines after the first blank line (common LLM separator).
    Returns None if the result is empty.
    """
    if not text:
        return None
    semi = text.find(";")
    if semi >= 0:
        result = text[: semi + 1].strip()
        return result or None
    # No semicolon: take lines up to the first blank line.
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            break
        kept.append(line)
    result = "\n".join(kept).strip()
    return result or None


class OpenRouterLLMClient:
    """LLM client using the OpenRouter SDK for chat completions."""

    provider_name = "openrouter"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        try:
            from openrouter import OpenRouter
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing dependency: install 'openrouter'.") from exc
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self._client = OpenRouter(api_key=api_key)

    def _chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> tuple[str, dict[str, int]]:
        """Send a chat request and return (content, per_call_stats).

        Stats are local to this call — no shared mutable state, making the
        method safe for concurrent or sequential use without cross-request mixing.
        """
        res = self._client.chat.send(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        choices = getattr(res, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenRouter response contained no choices.")
        raw_content = getattr(getattr(choices[0], "message", None), "content", None)

        # Handle both plain string and list-of-content-blocks (some model families).
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            text_parts: list[str] = []
            for block in raw_content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(str(block.text))
            content = " ".join(text_parts).strip()
            if not content:
                raise RuntimeError("OpenRouter response content blocks contained no text.")
        else:
            raise RuntimeError(
                f"OpenRouter response content has unexpected type: {type(raw_content)}"
            )

        # Token counting: prefer usage fields; fall back to word-count estimation.
        usage = getattr(res, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            if pt is not None and ct is not None:
                prompt_tokens = int(pt)
                completion_tokens = int(ct)
            else:
                est_pt = int(sum(len(m["content"].split()) * 4 // 3 for m in messages))
                est_ct = int(len(content.split()) * 4 // 3)
                prompt_tokens = int(pt) if pt is not None else est_pt
                completion_tokens = int(ct) if ct is not None else est_ct
        else:
            prompt_tokens = int(sum(len(m["content"].split()) * 4 // 3 for m in messages))
            completion_tokens = int(len(content.split()) * 4 // 3)

        call_stats: dict[str, int] = {
            "llm_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return content.strip(), call_stats

    @staticmethod
    def _extract_sql(text: str) -> str | None:
        """Extract a SQL statement from LLM output.

        Priority order:
        0. CANNOT_ANSWER sentinel — return None immediately.
        1. Iterate ALL fenced blocks:
           a. Accept ```sql-tagged blocks first.
           b. Then accept any fence whose body starts with SELECT or WITH.
           Reject fences tagged as text/plain/python/etc. that contain no SQL.
        2. Raw text fallback — find first SELECT/WITH keyword, trim trailing prose.
        """
        # Priority 0: explicit unanswerable sentinel.
        if re.search(r"\bCANNOT_ANSWER\b", text):
            return None

        # Priority 1: iterate all fenced blocks.
        sql_tagged: list[str] = []
        select_starting: list[str] = []
        for m in re.finditer(r"```(\w*)[^\n]*\n?(.*?)```", text, re.DOTALL):
            lang = m.group(1).lower().strip()
            body = m.group(2).strip()
            if not body:
                continue
            if lang == "sql":
                sql_tagged.append(body)
            elif re.match(r"(SELECT|WITH)\b", body, re.IGNORECASE):
                # Untagged or non-sql-tagged fence that contains SQL.
                select_starting.append(body)
            # Any other fence (```text, ```python, etc.) is ignored.

        for candidate in sql_tagged + select_starting:
            cleaned = _trim_sql_trailing_prose(candidate)
            if cleaned:
                return cleaned

        # Priority 2: raw text fallback.
        match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
        if match:
            fragment = text[match.start():]
            return _trim_sql_trailing_prose(fragment)

        return None

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        # Build schema block when schema info is available.
        schema_lines: list[str] = []
        if context.get("table") and context.get("columns"):
            schema_lines.append(f"Table: {context['table']}")
            schema_lines.append("Columns:")
            for col in context["columns"]:
                schema_lines.append(f"  - {col['name']} ({col['type']})")

        schema_block = "\n".join(schema_lines) if schema_lines else ""

        system_prompt = (
            "You are a SQLite SQL assistant. "
            "Generate a single read-only SELECT query from the natural language question "
            "using ONLY the provided columns. "
            "If the question asks about data that does not exist in the schema "
            "(e.g. columns not listed), respond with exactly: CANNOT_ANSWER\n"
            "Otherwise return ONLY the SQL query inside a markdown code fence "
            "(```sql ... ```). Do not include explanations. "
            "Do not invent or assume columns."
        )
        user_prompt = (
            (f"{schema_block}\n\n" if schema_block else "")
            + f"Question: {question}\n\n"
            "Generate the SQL query, or CANNOT_ANSWER if the question cannot be "
            "answered with these columns."
        )

        start = time.perf_counter()
        error: str | None = None
        sql: str | None = None
        accumulated = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            text, call_stats = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_tokens=4096,
            )
            for k in accumulated:
                accumulated[k] += call_stats[k]
            sql = self._extract_sql(text)
        except Exception as exc:
            error = str(exc)

        accumulated["model"] = self.model  # type: ignore[assignment]
        return SQLGenerationOutput(
            sql=sql,
            timing_ms=(time.perf_counter() - start) * 1000,
            llm_stats=accumulated,
            error=error,
        )

    def generate_answer(
        self,
        question: str,
        sql: str | None,
        rows: list[dict[str, Any]],
    ) -> AnswerGenerationOutput:
        _zero: dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": self.model,
        }

        if not sql:
            return AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=_zero,
                error=None,
            )

        if not rows:
            return AnswerGenerationOutput(
                answer="I cannot answer this question with the available data — the query returned no rows.",
                timing_ms=0.0,
                llm_stats=_zero,
                error=None,
            )

        # Scalar fast-path: single value — no LLM call needed.
        if len(rows) == 1 and len(rows[0]) == 1:
            value = next(iter(rows[0].values()))
            col = next(iter(rows[0].keys()))
            return AnswerGenerationOutput(
                answer=f"The result for '{col}' is: {value}",
                timing_ms=0.0,
                llm_stats=_zero,
                error=None,
            )

        system_prompt = (
            "You are a concise analytics assistant. "
            "Use only the provided SQL results. Do not invent data."
        )
        user_prompt = (
            f"Question:\n{question}\n\nSQL:\n{sql}\n\n"
            f"Rows (JSON):\n{json.dumps(rows[:30], ensure_ascii=True)}\n\n"
            "Write a concise answer in plain English."
        )

        start = time.perf_counter()
        error: str | None = None
        answer = ""
        accumulated = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            answer, call_stats = self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_tokens=2048,
            )
            for k in accumulated:
                accumulated[k] += call_stats[k]
        except Exception as exc:
            error = str(exc)
            answer = f"Error generating answer: {error}"

        accumulated["model"] = self.model  # type: ignore[assignment]
        return AnswerGenerationOutput(
            answer=answer,
            timing_ms=(time.perf_counter() - start) * 1000,
            llm_stats=accumulated,
            error=error,
        )

    def pop_stats(self) -> dict[str, Any]:
        """No-op kept for backwards compatibility. Stats are now per-call."""
        return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def build_default_llm_client() -> OpenRouterLLMClient:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")
    return OpenRouterLLMClient(api_key=api_key)
