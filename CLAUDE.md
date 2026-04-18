# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Discord bot ("Head of Archive" / Глава Архива) that answers questions about server history, treating logs as absolute truth. Russian-language persona-driven interactions. Runs locally via Ollama (qwen3:8b for reasoning, qwen3.5:4b for summarization, bge-m3 for embeddings).

## Commands

```bash
# Run the bot (starts Ollama + main.py)
start.bat
# or directly:
python main.py

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/unit/

# Run a single test file
pytest tests/unit/test_rag_cache.py

# Run prompt evaluation harness against a scenario
python tests/test_factory/run_harness.py --scenario tests/test_factory/scenarios/<scenario>.json
# With a draft prompt override:
python tests/test_factory/run_harness.py --scenario <scenario>.json --prompt tests/test_factory/.drafts/draft_prompt.txt --iteration 1

# Run standalone ingestion evaluation (mirrors production pipeline)
python tests/evaluate_prompts.py

# Prerequisites — Ollama models
ollama pull qwen3:8b
ollama pull bge-m3
```

## Architecture

Dual-agent ReAct system with strict separation of concerns:

**Agent2 (The Voice)** — `RAGAssistant.generate_refined_response()` in [run_llama_index.py](src/core/run_llama_index.py). Persona-driven conversational agent. Tools: `search_archive`, `peek_cached_searches`, `fetch_user_opinion`, `update_user_opinion`. Manages social memory (opinions) and delegates research to Agent1.

**Agent1 (The Researcher)** — `RAGAssistant.aquery()` in [run_llama_index.py](src/core/run_llama_index.py). Factual retrieval engine. Tools: `hybrid_search` (Vector + BM25 fusion → GPU rerank), `fetch_raw_logs` (neighbor expansion from SQLite), `execute_sql` (read-only SQL on messages table).

**ReAct Workflow** — [agent_core.py](src/core/agent_core.py). Custom `ReActAgentWorkflow` built on LlamaIndex's workflow engine. Injects Qwen3's native `<think>` blocks into LlamaIndex's ReAct `Thought:` parsing. Handles malformed LLM output with retry prompts.

**Ingestion Pipeline** — [ingestion.py](src/data/ingestion.py). ISO-week grouping → word-based token slicing (~750 words/chunk) → runt-chunk merging → 7-day continuity linking (prev/next_chunk_id) → sequential Ollama summarization → SQLite + Vector Store persistence. Uses `get_summarizer_llm()` with `keep_alive="120s"` warm-start for batch efficiency.

**RAG Cache** — [rag_cache.py](src/core/rag_cache.py). LRU cache (capacity=5) checked transparently by `search_archive` before invoking Agent1.

**GPU Reranker** — [dynamic_reranker.py](src/utils/dynamic_reranker.py). Loads FlagReranker to CPU, runs scoring on GPU, then immediately offloads (`to("cpu")` + `del` + `torch.cuda.empty_cache()`) to free ~1.2GB VRAM for Ollama.

**Message Flow**: Discord @mention → FIFO queue (`main.py`) → Agent2 → `search_archive` (cache miss) → Agent1 → hybrid_search → GPU rerank → fetch_raw_logs (neighbor expansion) → Agent1 report → Agent2 persona response.

**Storage**: Vector Store (summaries, `llama_index_storage/`), SQLite (raw messages, `discord_data.db`), JSON ledgers (cache/opinions/history in `cache/`).

## Logging

Three-tiered async logging via [logger_setup.py](src/utils/logger_setup.py). All entries carry `[TxID]` for cross-file correlation:

1. `agent_traces.log` (INFO) — ReAct steps, tool names, final answers
2. `agent_traces_detailed.log` (DEBUG) — full prompts, shadow thoughts, pre-rerank text
3. `agent_traces_full.log` (TRACE) — raw JSON I/O payloads

Additional logs: `system.log` (startup/ingestion), `chat_history.log` (user↔bot), `indexing_summarization.log` (ingest prompts/responses).

## Rules

- **New data storage paths** → add to `.gitignore`.
- **Prompt modifications** in `src/config/prompts.py` → use the prompt-engineering skill to respect model size constraints.
- **New functionality** must comply with SoC architecture (Agent1 = retrieval, Agent2 = persona/tools, ingestion = data pipeline).
- **No access** to `logs/`, `messages_json/`, `llama_index_storage/`, `cache/` — may contain PII.
- **Tests**: include in every implementation plan. For unplanned non-trivial logic changes, draft the implementation first, then ask before writing tests.
- **Dev Log**: append(to the file end) entries to `DEV_LOG.md` after every significant task. Format: datetime + exactly two sentences (What + Why).

## Key Constraints

- All LLM calls to Ollama are **sequential** (not concurrent) to avoid OOM on local GPU.
- SQLite I/O in async tools is offloaded via `asyncio.to_thread(run_with_context, ...)` to avoid blocking the event loop and preserve `transaction_id` propagation.
- `execute_sql` enforces read-only: regex blocks INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
- `fetch_raw_logs` has a 25K char ceiling with neighbor pruning to prevent context overflow.
- `None` values must never be stored in vector metadata — they serialize as the string `"None"` in some backends, poisoning neighbor lookups.
- `keep_alive=0` is the safe default for all LLM instances; only the ingestion summarizer overrides this (to `"120s"`) via `model_copy()` at the call site.
