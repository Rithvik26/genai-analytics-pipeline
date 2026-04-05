# Senior Full Stack Engineer (GenAI-Labs) Take-Home Assignment

## Timebox
Plan for **4-6 hours**.

## Goal
Optimize a baseline LLM-driven analytics pipeline for a single-table SQL dataset while preserving output quality.

Key metrics include **end-to-end response time** from prompt ingest to final answer, **resources consumed** (tokens), and **quality of the output**.

## Current Status

**This codebase is a starting point for the assignment and is not yet fully functional.** Several core components require implementation to make the pipeline production-ready:

- Token counting infrastructure (skeleton provided in `src/llm_client.py`, actual counting logic needs implementation)
- SQL validation and quality checks
- Result validation and answer quality verification
- Comprehensive observability (logging, metrics, tracing)
- Edge case handling and error recovery

The baseline pipeline will run, but key functionality—particularly around validation, observability, and efficiency optimizations—remains incomplete. See `Assignment Tasks` below and `CHECKLIST.md` for specific implementation requirements.

## What You Get
- Baseline Python pipeline with stages:
  - SQL generation (real LLM call)
  - SQL validation
  - SQL execution
  - Answer generation (real LLM call)
- Single SQLite table with gaming and mental health survey data
- Public tests and benchmark script
- OpenRouter integration via [OpenRouter Python SDK](https://pypi.org/project/openrouter/)
- Configurable model (default: `openai/gpt-5-nano`, override via `OPENROUTER_MODEL`)

## Assignment Tasks

1. **Make the system production-ready.** What does production-ready mean to you? Demonstrate whatever you consider essential.

2. **Ensure the system can generate accurate SQL queries.** The baseline may not work correctly out of the box. Identify what's missing and implement what's needed for reliable SQL generation.

3. **Maintain or improve answer correctness.** The system should handle edge cases gracefully.

4. **Design appropriate observability for this analytics pipeline.** Implement tracing, metrics, and logging as you see fit for production use.

5. **Implement a validation framework to ensure answer quality.** Consider SQL validation, result validation, and answer quality checks. (Hint: think about what SQL validation means in the context of an analytics pipeline.)

6. **Consider efficiency.** Optimize end-to-end latency, token usage, and efficient LLM requests while preserving quality.

## Hard Requirements
1. Do not modify existing public tests in `tests/test_public.py`.
2. Public tests must pass.
3. Keep the project runnable locally with standard Python.
4. Output contract: `AnalyticsPipeline.run()` must return a `PipelineOutput` instance, with each stage producing outputs that conform to the type schemas in `src/types.py`. This enables automated evaluation; submissions that deviate from it cannot be graded correctly.
5. Token counting must be implemented. The baseline includes a skeleton for tracking LLM usage statistics in `src/llm_client.py`, but you must implement the actual token counting. This is required for the efficiency evaluation to work.

## Production Readiness Requirements

Your submission **must include** a completed `CHECKLIST.md` file documenting your design decisions and implementation approach across all relevant areas.

## Requirements

- **Python:** 3.13+
- **Dependencies:** `openrouter`, `pandas` (see `requirements.txt`)

## Setup

### Data Setup

The dataset (~160MB) is not included in this repository. Download it before running the pipeline:

1. Go to [Kaggle - Gaming and Mental Health](https://www.kaggle.com/datasets/sharmajicoder/gaming-and-mental-health?select=gaming_mental_health_10M_40features.csv)
2. Download `gaming_mental_health_10M_40features.csv` (select this file from the dataset)
3. Place the file in the `data/` directory
4. **Important:** Ensure you download and use all 39 columns—do not drop any columns during download or import

The Kaggle page provides a more detailed description of the dataset, including column definitions and data sources.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/gaming_csv_to_db.py
python3 -m unittest discover -s tests -p "test_public.py"
```

### OpenRouter Setup

This project uses [OpenRouter](https://openrouter.ai/) to access LLMs for SQL generation and answer synthesis. OpenRouter provides a unified API for many models across providers. It offers a **free tier** that lets you use certain models at no cost, which is sufficient for this assignment.

To get started:

1. **Create an account** at [openrouter.ai](https://openrouter.ai/)
2. **Create an API key** in your account settings
3. **Set the API key** in your environment (or copy from `.env.example`):

```bash
set OPENROUTER_API_KEY=<your_key>
```

On Linux/macOS: `export OPENROUTER_API_KEY=<your_key>`

## Benchmark
Run:

```bash
python3 scripts/benchmark.py --runs 3
```

This prints baseline-style latency stats (`avg`, `p50`, `p95`) and success rate.

**Reference metrics** (baseline on reference hardware): avg ~2900ms, p50 ~2500ms, p95 ~4700ms, ~600 tokens/request. 

## Deliverables
1. Updated source code
2. Added tests (if any)
3. Completed `CHECKLIST.md` with all sections addressed
4. Short engineering note (`SOLUTION_NOTES.md`) with:
   - What you changed
   - Why you changed it
   - Measured impact (before/after benchmark numbers)
   - Tradeoffs and next steps

## Optional Part: Multi-Turn Conversation Support

This is an **optional** part for candidates who want to demonstrate additional capabilities. It is **not required** for a complete submission, but may contribute to bonus evaluation.

### The Problem

The current pipeline handles single, isolated questions. In real-world scenarios, users often ask follow-up questions that reference previous context:

- "What is the addiction level distribution by gender?"
- Follow-up: "What about males specifically?"
- Follow-up: "Can you explain the highest value?"
- Follow-up: "Now sort by anxiety score instead"

### Implementation Guidelines

- You may implement this however you see fit: extend the existing pipeline, add new modules, or integrate directly into the LLM client.
- No skeleton code or boilerplate is provided - design the solution architecture yourself.
- If implemented, document your approach in `CHECKLIST.md` under a "Follow-Up Questions" section.

## General Notes
- The baseline intentionally leaves room for substantial optimization.
- Hidden evaluation includes paraphrased prompts and edge/failure cases.
- Public tests are integration tests and require a valid `OPENROUTER_API_KEY`.
- Think beyond the obvious optimizations - the challenge tests your engineering judgment, not just your ability to follow a checklist.