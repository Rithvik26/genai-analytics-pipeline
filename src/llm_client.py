from __future__ import annotations

import json
import os
import time
from typing import Any

from src.types import SQLGenerationOutput, AnswerGenerationOutput

DEFAULT_MODEL = "openai/gpt-5-nano"


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
        self._stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _chat(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
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
        # Handle both plain string and list-of-content-blocks formats.
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            # Extract text from the first text block.
            text_parts = []
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
            raise RuntimeError(f"OpenRouter response content has unexpected type: {type(raw_content)}")

        # Token counting: prefer usage fields from the response; fall back to estimation.
        self._stats["llm_calls"] += 1
        usage = getattr(res, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            if pt is not None and ct is not None:
                self._stats["prompt_tokens"] += int(pt)
                self._stats["completion_tokens"] += int(ct)
                self._stats["total_tokens"] += int(pt) + int(ct)
            else:
                # Partial usage object — estimate what's missing.
                est_pt = int(sum(len(m["content"].split()) * 4 // 3 for m in messages))
                est_ct = int(len(content.split()) * 4 // 3)
                self._stats["prompt_tokens"] += int(pt) if pt is not None else est_pt
                self._stats["completion_tokens"] += int(ct) if ct is not None else est_ct
                self._stats["total_tokens"] += (
                    (int(pt) if pt is not None else est_pt)
                    + (int(ct) if ct is not None else est_ct)
                )
        else:
            # No usage object at all — estimate from token counts.
            est_pt = int(sum(len(m["content"].split()) * 4 // 3 for m in messages))
            est_ct = int(len(content.split()) * 4 // 3)
            self._stats["prompt_tokens"] += est_pt
            self._stats["completion_tokens"] += est_ct
            self._stats["total_tokens"] += est_pt + est_ct

        return content.strip()

    @staticmethod
    def _extract_sql(text: str) -> str | None:
        import re

        # Priority 0: explicit unanswerable sentinel from the model.
        if re.search(r"\bCANNOT_ANSWER\b", text):
            return None

        # Priority 1: markdown code fences  ```sql ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            if candidate:
                return candidate

        # Priority 2: JSON wrapper {"sql": "..."}
        maybe_json = text.strip()
        if maybe_json.startswith("{"):
            try:
                parsed = json.loads(maybe_json)
                sql = parsed.get("sql")
                if isinstance(sql, str) and sql.strip():
                    return sql.strip()
            except json.JSONDecodeError:
                pass

        # Priority 3: find first SELECT or WITH keyword (case-insensitive)
        match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
        if match:
            return text[match.start():].strip()

        return None

    def generate_sql(self, question: str, context: dict) -> SQLGenerationOutput:
        # Build schema block for the prompt when schema info is available.
        schema_lines: list[str] = []
        if context.get("table") and context.get("columns"):
            schema_lines.append(f"Table: {context['table']}")
            schema_lines.append("Columns:")
            for col in context["columns"]:
                schema_lines.append(f"  - {col['name']} ({col['type']})")

        schema_block = "\n".join(schema_lines) if schema_lines else ""

        system_prompt = (
            "You are a SQLite SQL assistant. "
            "Generate a single read-only SELECT query from the natural language question using ONLY the provided columns. "
            "If the question asks about data that does not exist in the schema (e.g. columns not listed), "
            "respond with exactly: CANNOT_ANSWER\n"
            "Otherwise return ONLY the SQL query inside a markdown code fence (```sql ... ```). "
            "Do not include explanations. Do not invent or assume columns."
        )
        user_prompt = (
            f"{schema_block}\n\n" if schema_block else ""
        ) + f"Question: {question}\n\nGenerate the SQL query, or CANNOT_ANSWER if the question cannot be answered with these columns."

        start = time.perf_counter()
        error = None
        sql = None

        try:
            text = self._chat(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=1.0,
                max_tokens=4096,
            )
            sql = self._extract_sql(text)
        except Exception as exc:
            error = str(exc)

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        return SQLGenerationOutput(
            sql=sql,
            timing_ms=timing_ms,
            llm_stats=llm_stats,
            error=error,
        )

    def generate_answer(
        self,
        question: str,
        sql: str | None,
        rows: list[dict[str, Any]],
        execution_error: str | None = None,
    ) -> AnswerGenerationOutput:
        _zero_stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": self.model}

        if not sql:
            return AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=_zero_stats,
                error=None,
            )

        if execution_error:
            return AnswerGenerationOutput(
                answer="I cannot answer this question with the available data.",
                timing_ms=0.0,
                llm_stats=_zero_stats,
                error=execution_error,
            )

        if not rows:
            return AnswerGenerationOutput(
                answer="I cannot answer this question with the available data — the query returned no rows.",
                timing_ms=0.0,
                llm_stats=_zero_stats,
                error=None,
            )

        # Fast path: single scalar result — no LLM call needed.
        if len(rows) == 1 and len(rows[0]) == 1:
            value = next(iter(rows[0].values()))
            col = next(iter(rows[0].keys()))
            return AnswerGenerationOutput(
                answer=f"The result for '{col}' is: {value}",
                timing_ms=0.0,
                llm_stats=_zero_stats,
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
        error = None
        answer = ""

        try:
            answer = self._chat(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=1.0,
                max_tokens=2048,
            )
        except Exception as exc:
            error = str(exc)
            answer = f"Error generating answer: {error}"

        timing_ms = (time.perf_counter() - start) * 1000
        llm_stats = self.pop_stats()
        llm_stats["model"] = self.model

        return AnswerGenerationOutput(
            answer=answer,
            timing_ms=timing_ms,
            llm_stats=llm_stats,
            error=error,
        )

    def pop_stats(self) -> dict[str, Any]:
        out = dict(self._stats or {})
        self._stats = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return out


def build_default_llm_client() -> OpenRouterLLMClient:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")
    return OpenRouterLLMClient(api_key=api_key)
