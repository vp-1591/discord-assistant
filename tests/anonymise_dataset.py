"""
anonymise_dataset.py
====================
Builds anonymized test chunks from real Discord message data.

Pipeline:
  1. Load & chunk messages using the same logic as evaluate_prompts.py
  2. Extract unique usernames/channels → build deterministic fake-name/fake-channel maps
  3. Apply name + channel maps via str.replace (no LLM needed)
  4. Send each chunk to qwen3.5:9b to DETECT remaining PII (addresses,
     phone numbers, real names, etc.) and return a translation table
  5. Apply detected PII translations on top
  6. Save anonymized results to tests/data/test_dataset.json
  7. Trace LLM calls (input, output, thinking, usage) to locally-hosted Langfuse

Run:  python tests/anonymise_dataset.py
"""

import os
import sys
import json
import re
import random
import asyncio
import datetime
import subprocess
from typing import List, Dict

# Load environment variables before Langfuse initialization
from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client, observe
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

# Langfuse client (reads LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL from env)
langfuse = get_client()

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.evaluate_prompts import (
    load_and_group_messages,
    build_chunks,
    MESSAGES_DIR,
    MIN_TOKEN_BYPASS,
    WORDS_PER_CHUNK,
)
from src.config.config import MESSAGES_DIR as CFG_MESSAGES_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_COUNT    = 30
RANDOM_SEED     = 45
OUTPUT_JSON     = os.path.join(ROOT, "tests", "data", "test_dataset.json")

# Realistic Russian first names for fake-name mapping (must be >= max unique
# users in the dataset).  Sorted for deterministic assignment via seed.
FAKE_NAMES = sorted([
    "Алексей", "Марина", "Дмитрий", "Елена", "Сергей",
    "Ольга", "Андрей", "Наталья", "Иван", "Татьяна",
    "Максим", "Анна", "Владимир", "Светлана", "Кирилл",
    "Юлия", "Павел", "Ирина", "Артём", "Екатерина",
    "Николай", "Виктория", "Роман", "Людмила", "Вячеслав",
    "Галина", "Борис", "Валентина", "Олег", "Тамара",
    "Евгений", "Лариса", "Денис", "Надежда", "Александр",
    "Полина", "Виктор", "Мария", "Григорий", "Дарья",
    "Степан", "Алиса", "Фёдор", "Вера", "Матвей",
    "Ксения", "Тимур", "Ева", "Лев", "Софья",
    "Михаил", "Варвара", "Пётр", "Зоя", "Василий",
    "Раиса", "Георгий", "Инна", "Леонид", "Жанна",
])

# Fake channel names (must be >= max unique channels).  Sorted for deterministic assignment.
FAKE_CHANNEL_NAMES = sorted([
    "альфа", "бета", "гамма", "дельта", "эпсилон",
    "дзета", "эта", "тета", "йота", "каппа",
])

# ---------------------------------------------------------------------------
# PII detection prompt — receives the list of already-applied fake names
# so the LLM never re-anonymizes them
# ---------------------------------------------------------------------------
PII_DETECT_PROMPT = """
##Context
{text}
##Task
Write a JSON object mapping only following PII types to an anonymized version.
PII types to map:
Email address, phone number, date of birth, IP addresses, names.

##Template
{{
  "pii_word1": "anonimised_word1",
  "pii_word2": "anonimised_word2",
  ...
}} 

##Example
Input:
JohnDoe: Hey, @DaveJohnson can you send the report to my email john@gmail.com
AnneSmith: I can do that for you.
Output:
{{
    "JohnDoe": "Mike",
    "DaveJohnson": "Chris",
    "AnneSmith": "Sara",
    "john@gmail.com": "mike@example.com"
}}
"""


# ---------------------------------------------------------------------------
# Step 1: Build deterministic name and channel maps
# ---------------------------------------------------------------------------

def build_name_map(chunks: List[dict]) -> dict:
    """Extract unique usernames from chunks and assign deterministic fake names."""
    seen_names = set()
    for chunk in chunks:
        for msg in chunk.get("messages", []):
            seen_names.add(msg["user"])

    sorted_names = sorted(seen_names)
    name_map = {}
    for i, real_name in enumerate(sorted_names):
        if i < len(FAKE_NAMES):
            name_map[real_name] = FAKE_NAMES[i]
        else:
            name_map[real_name] = f"Пользователь_{i + 1}"

    return name_map


def build_channel_map(chunks: List[dict]) -> dict:
    """Extract unique channel names from chunks and assign deterministic fake names (full replacement, not suffixing)."""
    seen_channels = set()
    for chunk in chunks:
        seen_channels.add(chunk["channel"])

    sorted_channels = sorted(seen_channels)
    channel_map = {}
    for i, real_ch in enumerate(sorted_channels):
        if i < len(FAKE_CHANNEL_NAMES):
            channel_map[real_ch] = FAKE_CHANNEL_NAMES[i]
        else:
            channel_map[real_ch] = f"канал_{i + 1}"

    return channel_map


# ---------------------------------------------------------------------------
# Step 2: Apply deterministic translations (names + channels)
# ---------------------------------------------------------------------------

def apply_deterministic_map(text: str, name_map: dict, channel_map: dict) -> str:
    """Replace nicknames and channel names using deterministic maps.

    Name map is now disabled - LLM handles author name anonymization.
    Only channel names are replaced deterministically here.
    """
    result = text

    # Name map disabled - LLM handles this
    # sorted_names = sorted(name_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    # for real_name, fake_name in sorted_names:
    #     result = result.replace(f"{real_name}:", f"@{fake_name}:")
    #     result = result.replace(f"@{real_name}", f"@{fake_name}")
    #     result = result.replace(real_name, fake_name)

    for real_ch, fake_ch in channel_map.items():
        result = result.replace(real_ch, fake_ch)

    return result


# ---------------------------------------------------------------------------
# Step 3: Detect remaining PII via LLM and build translation table
# ---------------------------------------------------------------------------

def parse_pii_table(raw_output: str) -> Dict[str, str]:
    """Extract the JSON translation table from the LLM response.

    The model may wrap JSON in markdown fences or add extra text —
    we strip everything outside the outermost {…}.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_output)
    cleaned = cleaned.strip()

    # Find the outermost JSON object
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return {}

    try:
        table = json.loads(match.group(0))
        if isinstance(table, dict):
            # Only keep string→string entries
            return {str(k): str(v) for k, v in table.items() if isinstance(v, str)}
    except json.JSONDecodeError:
        pass

    return {}


def apply_pii_table(text: str, pii_table: Dict[str, str]) -> str:
    """Apply PII translation table to text. Longer keys first."""
    result = text
    for original, replacement in sorted(pii_table.items(), key=lambda kv: len(kv[0]), reverse=True):
        result = result.replace(original, replacement)
    return result


# ---------------------------------------------------------------------------
# Step 3: Sequential LLM PII-detection pass
# ---------------------------------------------------------------------------

@observe(name="anonymise-chunks")
async def anonymise_chunks(
    chunks: List[dict],
    name_map: dict,
    channel_map: dict,
) -> List[dict]:
    llm = Ollama(
        model="qwen3.5:9b",
        request_timeout=450.0,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        context_window=8192,
        thinking=False,
        keep_alive="20s",
    )

    results = []
    pbar_total = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_result = {
            "chunk_id":      chunk["chunk_id"],
            "channel":       channel_map.get(chunk["channel"], chunk["channel"]),
            "date_range":    chunk["date"],
            "start_ts":      chunk["start_ts"],
            "end_ts":        chunk["end_ts"],
            "word_count":    len(chunk["text"].split()),
            "msg_count":    len(chunk.get("messages", [])),
            "bypass":        chunk.get("bypass", False),
            "anonymized_text": "",
                        "error":         None,
        }

        # Step 3a: Apply deterministic translations (names + channels)
        pre_anon = apply_deterministic_map(chunk["text"], name_map, channel_map)

        if chunk.get("bypass"):
            # Quiet-week bypass: deterministic maps are sufficient, no LLM call needed
            chunk_result["anonymized_text"] = pre_anon
            results.append(chunk_result)
            continue

        # Step 3b: Detect remaining PII via LLM
        fake_names_str = ", ".join(sorted(name_map.values()))
        prompt = PII_DETECT_PROMPT.format(fake_names=fake_names_str, text=pre_anon)
        messages = [ChatMessage(role="user", content=prompt)]

        with langfuse.start_as_current_observation(
            as_type="generation",
            name=f"detect-pii-chunk-{i+1}",
            model="qwen3.5:9b",
            input={
                "chunk_id": chunk["chunk_id"],
                "channel": chunk["channel"],
                "date": chunk["date"],
                "name_map_size": len(name_map),
                "prompt": prompt,
            },
        ) as gen_obs:
            try:
                res = await llm.achat(messages)
                content = res.message.content.strip() if res.message.content else ""

                # Extract thinking from response blocks (same pattern as agent_core.py)
                thinking_text = ""
                try:
                    blocks = getattr(res.message, "blocks", None)
                    if blocks is None:
                        blocks = res.message.additional_kwargs.get("blocks", [])
                    thinking_parts = [
                        b["content"] if isinstance(b, dict) else getattr(b, "content", "")
                        for b in (blocks or [])
                        if (b.get("block_type") if isinstance(b, dict) else getattr(b, "block_type", "")) == "thinking"
                    ]
                    thinking_text = "\n".join(thinking_parts)
                except Exception:
                    pass

                # Capture usage from Ollama raw response
                usage_details = {}
                metadata = {"thinking": thinking_text} if thinking_text else {}

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

                # Parse PII table from LLM output
                pii_table = parse_pii_table(content)
                
                # Step 3c: Apply PII translations on top of deterministic ones
                final_text = apply_pii_table(pre_anon, pii_table)
                chunk_result["anonymized_text"] = final_text

                gen_obs.update(
                    output=content,
                    usage_details=usage_details or None,
                    metadata=metadata or None,
                )

            except asyncio.CancelledError:
                chunk_result["anonymized_text"] = "[CANCELLED]"
                chunk_result["error"] = "Request cancelled (timeout or interrupt)"
                gen_obs.update(output="[CANCELLED]", metadata={"error": "cancelled"})
            except Exception as e:
                chunk_result["anonymized_text"] = f"[ERR] {type(e).__name__}: {e}"
                chunk_result["error"] = str(e)
                gen_obs.update(output=chunk_result["anonymized_text"], metadata={"error": str(e)})

        results.append(chunk_result)

        progress = f"[{i + 1}/{pbar_total}]"
        pii_count = len(chunk_result.get("pii_table", {}))
        status = "OK" if not chunk_result["error"] else f"ERR: {chunk_result['error']}"
        print(f"  {progress} channel={chunk['channel']} date={chunk['date']} — {status} (PII detected: {pii_count})")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(results: List[dict], total_available: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    output = {
        "metadata": {
            "generated_at":           now,
            "model":                  "qwen3.5:9b",
            "method":                 "pii_detection",
            "sample_size":             len(results),
            "seed":                    RANDOM_SEED,
            "total_chunks_available":  total_available,
            "words_per_chunk":         WORDS_PER_CHUNK,
            "min_token_bypass":        MIN_TOKEN_BYPASS,
        },
        "chunks": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@observe(name="anonymise-dataset-run")
async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Start ollama serve in background (no-op if already running)
    try:
        _ollama = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Started: ollama serve (pid={_ollama.pid})")
        await asyncio.sleep(2)
    except FileNotFoundError:
        print("[WARN] 'ollama' not found in PATH — assuming server is already running.")

    messages_dir = CFG_MESSAGES_DIR if os.path.exists(CFG_MESSAGES_DIR) else MESSAGES_DIR
    print(f"Messages directory : {messages_dir}")
    print(f"WORDS_PER_CHUNK    : {WORDS_PER_CHUNK}")
    print(f"MIN_TOKEN_BYPASS   : {MIN_TOKEN_BYPASS}")
    print(f"Sample count        : {SAMPLE_COUNT}")
    print(f"Random seed         : {RANDOM_SEED}")
    print()

    # Step 1: Build chunks
    groups = load_and_group_messages(messages_dir)
    print(f"Total (channel, week) groups: {len(groups)}")

    chunks = build_chunks(groups)
    bypass_count = sum(1 for c in chunks if c.get("bypass"))
    linked_count = sum(1 for c in chunks if c.get("prev_chunk_id") or c.get("next_chunk_id"))
    print(f"Total chunks produced       : {len(chunks)}")
    print(f"  Bypass (quiet weeks)       : {bypass_count}")
    print(f"  Continuity-linked          : {linked_count}")
    print()

    # Step 2: Sample and build deterministic maps
    random.seed(RANDOM_SEED)
    sample = random.sample(chunks, min(SAMPLE_COUNT, len(chunks)))
    print(f"Sampled {len(sample)} chunks (seed={RANDOM_SEED})")

    name_map = build_name_map(sample)
    channel_map = build_channel_map(sample)
    print(f"Name map entries    : {len(name_map)}")
    print(f"Channel map entries : {len(channel_map)}")

    # Step 3: Anonymize — deterministic maps first, then LLM PII detection
    print("Starting anonymization (deterministic maps + PII detection)...")
    results = await anonymise_chunks(sample, name_map, channel_map)

    # Step 4: Write output
    write_json(results, total_available=len(chunks), output_path=OUTPUT_JSON)

    langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())