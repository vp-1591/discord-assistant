"""
evaluate_prompts.py
===================
Standalone script to evaluate the ingestion summarization pipeline
against a random sample of real message chunks before a full run.

Run:  python tests/evaluate_prompts.py

Mirrors the production pipeline faithfully:
  - ISO-week circuit breaker grouping
  - Word-based sliding window chunking (WORDS_PER_CHUNK)
  - Runt-chunk merging + bypass for quiet weeks
  - 7-day continuity linking (prev/next chunk IDs)
  - get_summarizer_llm() (qwen3.5:4b) with INGESTION_SUMMARY_PROMPT
"""

import os
import sys
import json
import re
import random
import asyncio
import datetime
import uuid
import subprocess
from collections import defaultdict
from typing import List, Optional

# Load environment variables before Langfuse initialization
from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client, observe
from llama_index.core.llms import ChatMessage

# Langfuse client (reads LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL from env)
langfuse = get_client()

# ---------------------------------------------------------------------------
# Project Setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm

from src.config.config import (
    MESSAGES_DIR,
    LLM_CONTEXT_WINDOW,
    MAX_CHUNK_SIZE,
    get_summarizer_llm,
)
# Copied from src/config/prompts.py with test-specific constraints
INGESTION_SUMMARY_PROMPT = """SYSTEM: You are the Archivist of a Discord community. Your goal is to extract facts, decisions, and social dynamics for a RAG knowledge base.

CRITICAL INSTRUCTIONS:
1. CHRONOLOGICAL EXTRACTION: You must read the ENTIRE chunk day by day. Do NOT skip the beginning of the log. Extract key events chronologically to avoid information loss.
2. NO HALLUCINATIONS: Do not invent missing context. Understand context accurately (e.g., interpret timezones and generic gamer slang literally without creating false stories).
3. OUTPUT LANGUAGE: You MUST write the summary exactly in RUSSIAN, as the corpus is in Russian.

RULES:
1. STRUCTURE: [Tag] NAME: FACT/EVENT. (Use tags: [Game], [Lore], [Social], [Meta])
2. PRIORITY: Server lore (e.g., leadership titles, role shifts), key decisions, plans, game outcomes, and technical discussions.
3. SOCIAL CONTEXT: Conflicts, roles, and inside jokes are valid social facts.
4. FORMAT: Continuous text. No introductory fluff. No more than number of days*2 sentences. Keep it concise but do not omit days.

Example:
Input:
--- 2023-07-24 ---
Alice: Я проиграла спор, поэтому Bob теперь лидер
--- 2023-07-25 ---
Charlie: го в кооп вечером
Output:
[Lore] Alice: Заявила, что проиграла спор, и теперь Bob становится лидером. [Game] Charlie: Предложил поиграть в кооператив вечером.

ДАННЫЕ:
{chunk}
"""

# ---------------------------------------------------------------------------
# Constants
# NOTE: WORDS_PER_CHUNK is intentionally smaller here than in ingestion.py
#       to keep test chunks at ~1000 tokens (700 words @ ~1.3 tok/word for Russian).
# ---------------------------------------------------------------------------
SAMPLE_COUNT    = 5
RANDOM_SEED     = 46
OUTPUT_PATH     = os.path.join(ROOT, "logs", "prompt_evaluation_results.md")

MIN_TOKEN_BYPASS = 50    # quiet-week threshold (words)
WORDS_PER_CHUNK  = 700   # ~1000 tokens — TEST OVERRIDE (production uses 1536)

# ---------------------------------------------------------------------------
# Mention & noise filtering  (mirrors ingestion._read_and_group_messages)
# ---------------------------------------------------------------------------
_mention_re      = re.compile(r'<@(!|&)?(\d+)>')
_link_re         = re.compile(r'https?://\S+')
_inline_code_re  = re.compile(r'\b(?=.*\d)[A-Za-z0-9]{3,6}-[A-Za-z0-9]{3,6}\b')
_noise_re        = re.compile(
    r'-*[a-fA-F0-9]{32,128}-*|'
    r'-*[a-fA-F0-9]{8,12}-*|'
    r'-+[a-fA-F0-9]{4,12}-+|'
    r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
)


def _resolve_mentions(text: str, fallback_map: dict) -> str:
    def replace_match(m):
        name = fallback_map.get(m.group(2))
        return f"@{name}" if name else m.group(0)
    return _mention_re.sub(replace_match, text)


# ---------------------------------------------------------------------------
# Step 1 & 2: Read JSON files → group by (channel, ISO-week)
# ---------------------------------------------------------------------------

def load_and_group_messages(messages_dir: str) -> dict:
    if not os.path.exists(messages_dir):
        raise FileNotFoundError(f"messages_json directory not found: {messages_dir}")

    json_files = [f for f in os.listdir(messages_dir) if f.endswith(".json")]

    # Pass 1 — build the global name fallback map
    fallback_map: dict = {}
    for filename in tqdm(json_files, desc="🔖 Building name map"):
        path = os.path.join(messages_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for msg in data:
                    fallback_map.update(msg.get("last_known_names", {}))
        except Exception:
            pass

    # Pass 2 — read, filter, and group by (channel, ISO-week)
    groups: dict = defaultdict(list)   # (channel, week_str) → list of message dicts

    for filename in tqdm(json_files, desc="📂 Reading JSON history"):
        path = os.path.join(messages_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue

            last_user, last_msg_text = None, None
            for msg in data:
                raw_text = str(msg.get("message", "")).strip()
                if not raw_text:
                    continue

                user_id     = str(msg.get("user_id", "Unknown"))
                author_name = fallback_map.get(user_id, f"User_{user_id}")

                text = _resolve_mentions(raw_text, fallback_map)
                text = _link_re.sub("", text)
                text = _inline_code_re.sub("", text)
                text = _noise_re.sub("", text).strip()
                if not text:
                    continue

                # De-duplicate consecutive identical messages
                if author_name == last_user and text == last_msg_text:
                    continue
                last_user, last_msg_text = author_name, text

                ts       = msg.get("timestamp", "2000-01-01T00:00:00")
                date_str = ts.split("T")[0]

                # ISO-week circuit breaker (matches production logic)
                try:
                    dt  = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    iso = dt.isocalendar()
                    week_str = f"{iso.year}-W{iso.week:02d}"
                except ValueError:
                    week_str = date_str

                channel = msg.get("channel", "unknown")
                groups[(channel, week_str)].append({
                    "id":        str(msg.get("message_id", uuid.uuid4())),
                    "user":      author_name,
                    "text":      text,
                    "date":      date_str,
                    "channel":   channel,
                    "timestamp": ts,
                })

        except Exception:
            pass

    return groups


# ---------------------------------------------------------------------------
# Step 3 & 4: Build word-bounded chunks with continuity linking
#             Mirrors ingestion._build_raw_chunks exactly
# ---------------------------------------------------------------------------

def build_chunks(groups: dict) -> List[dict]:
    final_chunks = []

    # Re-group by channel → sorted list of (week_str, messages)
    channel_groups: dict = defaultdict(list)
    for (channel, week_str), msgs in groups.items():
        channel_groups[channel].append((week_str, msgs))

    for channel, week_data in tqdm(channel_groups.items(), desc="✂️  Chunking by week"):
        week_data.sort(key=lambda x: x[0])   # chronological weeks

        channel_chunks: List[dict] = []

        for week_str, messages in week_data:
            messages.sort(key=lambda x: x["timestamp"])

            current_batch: list = []
            current_words: int  = 0
            week_level_chunks: list = []

            for m in messages:
                msg_words = len(m["text"].split())
                if (current_words + msg_words > WORDS_PER_CHUNK) and current_batch:
                    week_level_chunks.append(current_batch)
                    current_batch = []
                    current_words = 0
                current_batch.append(m)
                current_words += msg_words

            if current_batch:
                # Merge runt chunk with previous if it's too small
                if current_words < MIN_TOKEN_BYPASS and week_level_chunks:
                    week_level_chunks[-1].extend(current_batch)
                else:
                    week_level_chunks.append(current_batch)

            total_week_words = sum(len(m["text"].split()) for m in messages)
            is_quiet_week    = total_week_words < MIN_TOKEN_BYPASS

            for wk_msgs in week_level_chunks:
                # Build text with day-marker separators
                text_lines = []
                last_day   = None
                for m in wk_msgs:
                    if m["date"] != last_day:
                        text_lines.append(f"\n--- {m['date']} ---")
                        last_day = m["date"]
                    text_lines.append(f"{m['user']}: {m['text']}")

                chunk_text = "\n".join(text_lines).strip()
                channel_chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text":     chunk_text,
                    "date":     week_str,
                    "channel":  channel,
                    "messages": wk_msgs,
                    "bypass":   is_quiet_week,
                    "start_ts": wk_msgs[0]["timestamp"],
                    "end_ts":   wk_msgs[-1]["timestamp"],
                })

        # 7-day continuity linking
        for i, chunk in enumerate(channel_chunks):
            if i > 0:
                prev = channel_chunks[i - 1]
                t1   = datetime.datetime.fromisoformat(prev["end_ts"].replace("Z", "+00:00"))
                t2   = datetime.datetime.fromisoformat(chunk["start_ts"].replace("Z", "+00:00"))
                if (t2 - t1).days <= 7:
                    chunk["prev_chunk_id"] = prev["chunk_id"]

            if i < len(channel_chunks) - 1:
                nxt = channel_chunks[i + 1]
                t1  = datetime.datetime.fromisoformat(chunk["end_ts"].replace("Z", "+00:00"))
                t2  = datetime.datetime.fromisoformat(nxt["start_ts"].replace("Z", "+00:00"))
                if (t2 - t1).days <= 7:
                    chunk["next_chunk_id"] = nxt["chunk_id"]

        final_chunks.extend(channel_chunks)

    return final_chunks


# ---------------------------------------------------------------------------
# Step 5: Sequential Ollama summarization
#         Uses get_summarizer_llm() + INGESTION_SUMMARY_PROMPT — same as production
# ---------------------------------------------------------------------------

@observe(name="evaluate-summarization")
async def evaluate(chunks: List[dict]) -> List[dict]:
    results   = []
    _base_llm = get_summarizer_llm()
    llm       = _base_llm.model_copy(update={"keep_alive": "20s"})

    pbar = tqdm(total=len(chunks), desc="🤖 Summarising with Ollama")

    for i, chunk in enumerate(chunks):
        summary: str
        thinking: str  = ""
        raw_debug: str = ""

        if chunk.get("bypass"):
            summary = "[BYPASS] Quiet week — stored as RAW text (no LLM call)."
        else:
            prompt = INGESTION_SUMMARY_PROMPT.format(chunk=chunk["text"])
            messages = [ChatMessage(role="user", content=prompt)]

            with langfuse.start_as_current_observation(
                as_type="generation",
                name=f"summarize-chunk-{i+1}",
                model="qwen3.5:4b",
                input={
                    "chunk_id": chunk["chunk_id"],
                    "channel": chunk["channel"],
                    "date": chunk["date"],
                    "word_count": len(chunk["text"].split()),
                    "prompt": prompt,
                },
            ) as gen_obs:
                try:
                    res = await llm.achat(messages)
                    summary = res.message.content.strip()

                    # Extract thinking from response blocks (same pattern as agent_core.py)
                    try:
                        blocks = getattr(res.message, "blocks", None)
                        if blocks is None:
                            blocks = res.message.additional_kwargs.get("blocks", [])
                        thinking_parts = [
                            b["content"] if isinstance(b, dict) else getattr(b, "content", "")
                            for b in (blocks or [])
                            if (b.get("block_type") if isinstance(b, dict) else getattr(b, "block_type", "")) == "thinking"
                        ]
                        thinking = "\n".join(thinking_parts)
                    except Exception:
                        pass

                    # Capture usage and timing from Ollama raw response
                    usage_details = {}
                    metadata = {"thinking": thinking} if thinking else {}

                    if hasattr(res, "raw") and isinstance(res.raw, dict):
                        raw = res.raw
                        prompt_tokens = raw.get("prompt_eval_count", 0)
                        completion_tokens = raw.get("eval_count", 0)
                        usage_details = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        }
                        total_dur = raw.get("total_duration", 0) / 1e9
                        load_dur = raw.get("load_duration", 0) / 1e9
                        eval_dur = raw.get("eval_duration", 0) / 1e9
                        metadata["timing"] = {
                            "total_s": round(total_dur, 2),
                            "load_s": round(load_dur, 2),
                            "gen_s": round(eval_dur, 2),
                            "tok_per_s": round(completion_tokens / max(1, eval_dur), 1),
                        }

                    gen_obs.update(
                        output=summary,
                        usage_details=usage_details or None,
                        metadata=metadata or None,
                    )

                    if not summary:
                        raw_debug = (
                            f"content='{res.message.content}'\n"
                            f"thinking='{thinking}'\n"
                            f"additional_kwargs={res.message.additional_kwargs}\n"
                            f"raw={getattr(res, 'raw', 'N/A')}"
                        )

                        print(f"\n[WARN] Chunk {i+1}: LLM returned empty content.")
                        summary = "[EMPTY RESPONSE — see raw_debug field]"

                except asyncio.CancelledError:
                    summary   = "[CANCELLED] Request was cancelled (timeout or interrupt)."
                    raw_debug = "asyncio.CancelledError"
                    print(f"[WARN] Chunk {i+1}: request cancelled.")
                    gen_obs.update(output=summary, metadata={"error": "cancelled"})
                except Exception as e:
                    summary   = f"[ERR] {type(e).__name__}: {e}"
                    raw_debug = repr(e)
                    print(f"[WARN] Chunk {i+1}: exception — {e}")
                    gen_obs.update(output=summary, metadata={"error": str(e)})

        results.append({
            "index":         i + 1,
            "chunk_id":      chunk["chunk_id"],
            "channel":       chunk["channel"],
            "date":          chunk["date"],
            "start_ts":      chunk["start_ts"],
            "end_ts":        chunk["end_ts"],
            "bypass":        chunk.get("bypass", False),
            "prev_chunk_id": chunk.get("prev_chunk_id"),
            "next_chunk_id": chunk.get("next_chunk_id"),
            "word_count":    len(chunk["text"].split()),
            "msg_count":     len(chunk["messages"]),
            "text":          chunk["text"],
            "summary":       summary,
            "thinking":      thinking,
            "raw_debug":     raw_debug,
        })

        pbar.update(1)

    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_markdown(results: List[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Ingestion Pipeline — Prompt Evaluation\n",
        f"**Generated:** {now}",
        f"**Chunks evaluated:** {len(results)}",
        f"**WORDS_PER_CHUNK:** {WORDS_PER_CHUNK} | **MIN_TOKEN_BYPASS:** {MIN_TOKEN_BYPASS}",
        f"**Summarizer model:** qwen3.5:4b (via `get_summarizer_llm()`)\n",
        "---\n",
    ]

    for r in results:
        bypass_flag = " ⚡ BYPASS" if r["bypass"] else ""
        lines.append(f"## Chunk {r['index']}{bypass_flag} — `{r['channel']}` / {r['date']}")
        lines.append(f"- **chunk_id:** `{r['chunk_id']}`")
        lines.append(f"- **Period:** {r['start_ts']} → {r['end_ts']}")
        lines.append(f"- **Messages:** {r['msg_count']} | **Words:** {r['word_count']}")

        if r["prev_chunk_id"]:
            lines.append(f"- **← prev_chunk_id:** `{r['prev_chunk_id']}`")
        if r["next_chunk_id"]:
            lines.append(f"- **→ next_chunk_id:** `{r['next_chunk_id']}`")

        lines.append("")
        lines.append("### 📄 Source Text\n```")
        lines.append(r["text"])
        lines.append("```\n")

        if r.get("thinking"):
            lines.append("### Thinking Process")
            lines.append(f"> {r['thinking'].replace(chr(10), chr(10) + '> ')}\n")

        lines.append("### INGESTION_SUMMARY_PROMPT output")
        lines.append(r["summary"] if r["summary"] else "_(empty)_")

        if r.get("raw_debug"):
            lines.append("\n### Raw Debug Dump")
            lines.append(f"```\n{r['raw_debug']}\n```")

        lines.append("\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Results written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@observe(name="evaluate-prompts-run")
async def main() -> None:
    # Ensure stdout is UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Start ollama serve in background (no-op if already running)
    try:
        _ollama = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Started: ollama serve (pid={_ollama.pid})")
        await asyncio.sleep(2)  # give it a moment to bind
    except FileNotFoundError:
        print("[WARN] 'ollama' not found in PATH — assuming server is already running.")

    print(f"Messages directory : {MESSAGES_DIR}")
    print(f"WORDS_PER_CHUNK    : {WORDS_PER_CHUNK}  (~1000 tokens, test override)")
    print(f"MIN_TOKEN_BYPASS   : {MIN_TOKEN_BYPASS}")
    print()

    groups = load_and_group_messages(MESSAGES_DIR)
    print(f"Total (channel, week) groups : {len(groups)}")

    chunks = build_chunks(groups)
    bypass_count = sum(1 for c in chunks if c.get("bypass"))
    linked_count = sum(1 for c in chunks if c.get("prev_chunk_id") or c.get("next_chunk_id"))
    print(f"Total chunks produced        : {len(chunks)}")
    print(f"  Bypass (quiet weeks)       : {bypass_count}")
    print(f"  Continuity-linked          : {linked_count}")
    print()

    random.seed(RANDOM_SEED)
    sample = random.sample(chunks, SAMPLE_COUNT) if len(chunks) > SAMPLE_COUNT else chunks
    print(f"Sampled {len(sample)} chunks (seed={RANDOM_SEED})")
    print()

    results = await evaluate(sample)
    write_markdown(results, OUTPUT_PATH)

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
