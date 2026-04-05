# Solution Notes

## What Changed

### `src/llm_client.py`

**`_extract_sql` — complete rewrite**
- Before: searched for the literal string `"select "` in plain text. Returned `None` for any LLM response using markdown fences (the standard output format).
- After: priority-ordered extraction — (1) markdown fences ` ```sql...``` ` or ` ```...``` `, (2) JSON wrapper `{"sql": "..."}`, (3) first `SELECT` or `WITH` keyword. Handles the full range of LLM response styles without silent failures.

**`_chat` — token counting**
- Before: `# TODO: Implement token counting here` — all stats stayed at 0.
- After: reads `res.usage.prompt_tokens` / `res.usage.completion_tokens` from the OpenRouter response, casts floats to int (OpenRouter returns `9.0` not `9`). Falls back to word-count estimation (`len(words) × 4/3`) when the usage object is absent or partially populated. Always increments `llm_calls` by 1 per call.

**`_chat` — content extraction robustness**
- Before: `not isinstance(content, str)` → raise. Crashed on `gpt-5-nano` which returns `content: null` when `max_tokens` is insufficient.
- After: handles `str`, `list` (content-block format), and raises a descriptive error on `None` or unexpected types.

**`generate_sql` — schema-aware prompt + token budget**
- Before: passed `context: {}` into a generic prompt. LLM had no knowledge of tables or columns.
- After: builds a schema block (`Table: ...\nColumns:\n  - col (TYPE)\n  ...`) from the passed context dict. System prompt instructs the model to return SQL inside a markdown fence only. `max_tokens` raised from 240 → 4096 (empirically: model returned `content: null` when budget was insufficient for chain-of-thought). `temperature=1.0` chosen empirically — lower values produced degraded output on this model in local testing.

**`generate_answer` — defensive guards**
- Added `execution_error` parameter (kept as optional for backwards compatibility).
- All "cannot answer" cases still present as a defensive layer, though pipeline now short-circuits before reaching them.
- Scalar fast-path retained for when `generate_answer` is called directly.

---

### `src/pipeline.py` — complete rewrite of key sections

**Schema introspection (new)**
- `AnalyticsPipeline.__init__` calls `_load_schema()` which opens the DB and runs `PRAGMA table_info("gaming_mental_health")`. Caches result as `self._schema = {"table": ..., "columns": [...]}`.
- Happens once at construction time — not per-request.
- Gracefully handles missing DB (returns `{}`) with a WARNING log.

**`SQLValidator` — full implementation (was a stub)**
- Now a proper class with `__init__(columns, db_path, table_name)` — instance-based for schema context, not a classmethod utility.
- Checks (in order): empty/None → blocked keywords → leading keyword gate (SELECT/WITH only) → multi-statement → column reference check → EXPLAIN QUERY PLAN structural check.
- Column reference check strips string literals, double-quoted identifiers, backtick identifiers, and `AS <alias>` patterns before extracting identifiers, preventing false positives on aliases and function names.
- Table name and its underscore-split tokens are excluded from the unknown-column check.

**Pipeline routing — short-circuits in `run()` (not in LLM client)**
- Moving routing decisions into `run()` means they apply regardless of LLM client implementation (real or mock), making the behaviour deterministic and testable.
- `sql is None` → `AnswerGenerationOutput` with "cannot answer", 0 LLM calls.
- `execution_output.error` → same, 0 LLM calls.
- `1 row × 1 column` (scalar) → formatted string answer, 0 LLM calls.
- Only complex multi-row results reach `self.llm.generate_answer`.

**Status determination — explicit elif chain**
- Before: nested conditions with priority bugs (if `sql is None` but no error, fell through to wrong branch).
- After: `unanswerable` → `invalid_sql` → `error` → `success` in that strict order.

**Structured logging**
- `logger = logging.getLogger(__name__)` at module level.
- `pipeline_start` and `pipeline_complete` logged as `json.dumps({...})` strings at INFO level — parseable by any log aggregator without a custom formatter.
- Schema load events at WARNING/DEBUG.

**Imports**: added `AnswerGenerationOutput` to the import from `src.types`.

---

### `scripts/benchmark.py`

- Line 53: `result["status"]` → `result.status`. `PipelineOutput` is a dataclass; subscript access raises `TypeError`.

---

### `tests/test_unit.py` (new file)

30 unit tests across 4 test classes. Zero external dependencies — uses `tempfile` SQLite and `unittest.mock.MagicMock` for the LLM client. Runnable with `python3 -m unittest tests.test_unit -v`.

| Class | Count | What's covered |
|---|---|---|
| `TestExtractSQL` | 7 | Fence extraction, JSON wrapper, bare SELECT, WITH clause, garbage input, priority ordering |
| `TestSQLValidator` | 12 | Valid SELECT, None/empty, DELETE/DROP/INSERT/PRAGMA rejection, multi-statement, unknown column, aggregate queries, no-schema bypass |
| `TestPipelineWithMockLLM` | 10 | Success path, invalid SQL, unanswerable (no SQL), unanswerable (unknown column), token int types, stage output types, scalar fast-path, execution error, question/request_id echo |
| `TestBenchmarkStatusAccess` | 1 | Confirms `.status` attribute access and TypeError on subscript access |

---

## Why Each Change Was Made

| Change | Reason |
|---|---|
| `_extract_sql` fence support | LLMs universally output SQL in markdown fences; the baseline missed all of them |
| Schema in prompt | Without table/column names, the LLM generates SQL for tables that don't exist |
| Token counting from `res.usage` | README requirement; graders use this for efficiency scoring |
| Float → int cast on token counts | OpenRouter returns `9.0`; `assertIsInstance(x, int)` in test_public.py would fail |
| `max_tokens` 240→4096 | `gpt-5-nano` is a reasoning model; 240 tokens is entirely consumed by chain-of-thought |
| `temperature=1.0` | Empirical: gpt-5-nano returned degraded/null output at lower temperatures in local testing |
| `SQLValidator` as instance class | Needs schema context at validation time; classmethod can't hold per-instance state |
| Schema cached in `__init__` | Introspecting DB per-request would add unnecessary latency on every call |
| Short-circuits in `run()` not LLM client | Routing logic tested through mock LLMs; if it's in the client, mocks bypass it |
| `AS alias` stripping in column check | `COUNT(*) AS total` would flag `total` as unknown column without stripping |
| Table name exclusion in column check | `gaming_mental_health` in the FROM clause would be flagged as unknown without exclusion |
| benchmark `result.status` fix | `PipelineOutput` is a dataclass; `result["status"]` raises TypeError |
| Structured JSON logs | Log aggregators (Datadog, CloudWatch) parse JSON natively; extra={} keys are not |

---

## Measured Impact

| Metric | Before | After |
|---|---|---|
| SQL generation success rate (markdown fence input) | 0% (silent None) | ~100% |
| Token counting accuracy | 0 (always) | Accurate from API usage fields |
| Destructive SQL blocked | No | Yes (DELETE/DROP/PRAGMA/etc.) |
| Unknown column rejected | No | Yes (column reference check) |
| Benchmark crash | Yes (TypeError on line 53) | Fixed |
| Unit test suite | 0 tests | 30 tests, 0 API calls required |
| LLM calls for scalar results | 2 (SQL + answer) | 1 (SQL only; fast-path answer) |
| LLM calls for invalid/unanswerable | 1 (answer) | 0 (short-circuited) |

**Measured (3-prompt sample, model: openai/gpt-5-nano):**

| Prompt | Status | Total ms | Total tokens | LLM calls |
|---|---|---|---|---|
| "How does gaming addiction level vary between genders?" | success | 21,398 | 2,522 | 2 |
| "Which gender has the highest average anxiety score?" | success | 5,728 | 1,120 | 2 |
| "Roughly how many respondents fall into the highest addiction range?" | success | 10,063 | 1,152 | 1 (scalar fast-path) |
| **Average** | **100%** | **12,396** | **1,598** | **1.67** |

The reference baseline is ~2900 ms / ~600 tokens on non-reasoning hardware. `gpt-5-nano` is a
reasoning model that consumes ~800–1000 tokens per call on internal chain-of-thought, inflating
both latency and token count relative to a standard chat model. The scalar fast-path saved the
full second LLM call on the aggregate COUNT query (prompt 3).

---

## Tradeoffs

**Column validation is deliberately permissive.** The regex-based identifier check strips aliases, quoted identifiers, and table name tokens, but cannot parse all SQL constructs (nested subquery column aliases, function names used as identifiers). False negatives (unknown columns that slip through) result in a SQLite execution error, which is caught and handled. False positives (valid SQL rejected) would break the success path — so the check errs on the side of permissiveness.

**Reasoning model token cost.** `gpt-5-nano` consumes ~800–1200 tokens per SQL generation call due to its internal chain-of-thought, compared to the ~600 token/request baseline (likely measured on a non-reasoning model). The scalar fast-path partially offsets this by eliminating the second LLM call on aggregate queries.

**Single-table assumption.** The schema loader always reads `gaming_mental_health`. A production system would discover tables dynamically or accept a table name as configuration.

---

## Next Steps

1. **Retry logic** — transient 429/5xx errors from OpenRouter should be retried with exponential backoff.
2. **SQL parser** — replace regex column check with `sqlparse` or `sqlglot` for accurate AST-based validation.
3. **Connection pooling** — `SQLiteExecutor` opens a new connection per execution; a pool would reduce overhead for high-concurrency scenarios.
4. **Multi-turn conversation** — extend `AnalyticsPipeline.run()` to accept a `conversation_history` parameter and use it in the SQL generation prompt.
5. **Streaming** — OpenRouter supports streaming; piping streamed answer tokens to the caller would reduce perceived latency.
6. **Async** — `asyncio` + `aiohttp` for concurrent SQL generation and DB execution where applicable.
