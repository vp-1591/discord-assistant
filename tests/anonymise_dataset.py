"""
anonymise_dataset.py
====================
Builds anonymized test chunks from real Discord message data.

Pipeline:
  1. Load & chunk messages using the same logic as evaluate_prompts.py
  2. Extract unique usernames → build deterministic fake-name map
  3. Send each chunk to qwen3.5:9b (with thinking) for anonymization
  4. Save anonymized results to tests/data/test_dataset.json
  5. Log original text + thinking + result to logs/temp_anon_trace.log

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
from typing import List

from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

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
SAMPLE_COUNT    = 1
RANDOM_SEED     = 46
OUTPUT_JSON     = os.path.join(ROOT, "tests", "data", "test_dataset.json")
TRACE_LOG_PATH  = os.path.join(ROOT, "logs", "temp_anon_trace.log")

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

# ---------------------------------------------------------------------------
# Anonymization prompt
# ---------------------------------------------------------------------------
ANONYMIZATION_PROMPT = """СИСТЕМА: Ты — модуль деидентификации данных. Твоя задача — заменить все персональные данные в тексте Discord-переписки на реалистичные подстановки, сохранив структуру, смысл и социальную динамику.

ПРАВИЛА:
1. ИМЕНА: Замени каждое реальное имя на указанное в NAME_MAP ниже. Имя должно быть заменено ВЕЗДЕ, где оно появляется — как в префиксе "Имя:", так и внутри текста сообщений.
2. АДРЕСА: Замени реальные адреса (улицы, номера домов, города) на реалистичные российские адреса, которых не существует.
3. ТЕЛЕФОНЫ: Замени номера телефонов на реалистичные российские номера в формате +7-XXX-XXX-XX-XX, которых не существует.
4. СООТНОШЕНИЯ: Критически важно сохранять СООТНОШЕНИЯ между людьми. Если в оригинале два человека — друзья, враги, родственники — это должно остаться. Один и тот же человек ВСЕГДА получает одно и то же подстановочное имя.
5. СТРУКТУРА: Сохрани формат дневных разделителей (--- дата ---), порядок сообщений, даты, времена и общую структуру текста.
6. ЯЗЫК: Выход должен быть на том же языке, что и вход (русский).
7. ФОРМАТ: Выведи ТОЛЬКО обезличенный текст. Никаких комментариев, пояснений, мета-информации или JSON-обёрток.

NAME_MAP:
{name_map}

ДАННЫЕ:
{chunk}"""


# ---------------------------------------------------------------------------
# Step 1: Build deterministic name map
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
            # Fallback: generate numbered names if we have more users than our list
            name_map[real_name] = f"Пользователь_{i + 1}"

    return name_map


# ---------------------------------------------------------------------------
# Step 2: Sequential LLM anonymization
# ---------------------------------------------------------------------------

async def anonymise_chunks(chunks: List[dict], name_map: dict) -> List[dict]:
    llm = Ollama(
        model="qwen3.5:9b",
        request_timeout=450.0,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        context_window=8192,
        thinking=True,
        keep_alive="20s",
    )

    # Format name map for the prompt
    name_map_str = "\n".join(f"  {real} → {fake}" for real, fake in sorted(name_map.items()))

    results = []
    pbar_total = len(chunks)

    for i, chunk in enumerate(chunks):
        chunk_result = {
            "chunk_id":      chunk["chunk_id"],
            "channel":       chunk["channel"],
            "date_range":    chunk["date"],
            "start_ts":      chunk["start_ts"],
            "end_ts":        chunk["end_ts"],
            "word_count":    len(chunk["text"].split()),
            "msg_count":    len(chunk.get("messages", [])),
            "bypass":        chunk.get("bypass", False),
            "anonymized_text": "",
            "error":         None,
        }

        if chunk.get("bypass"):
            # Quiet-week bypass: simple string replacement, no LLM call
            anon_text = chunk["text"]
            for real_name, fake_name in name_map.items():
                # Replace in "RealName: message" prefixes and within text
                anon_text = anon_text.replace(f"{real_name}:", f"{fake_name}:")
                anon_text = anon_text.replace(f"{real_name}", f"{fake_name}")
            chunk_result["anonymized_text"] = anon_text
            results.append(chunk_result)
            continue

        prompt = ANONYMIZATION_PROMPT.format(
            name_map=name_map_str,
            chunk=chunk["text"],
        )
        messages = [ChatMessage(role="user", content=prompt)]

        thinking_text = ""
        raw_debug = ""

        try:
            res = await llm.achat(messages)
            content = res.message.content.strip() if res.message.content else ""
            thinking_text = getattr(res.message, "thinking", "") or ""

            if content:
                chunk_result["anonymized_text"] = content
            else:
                raw_debug = (
                    f"content='{res.message.content}'\n"
                    f"thinking='{thinking_text}'\n"
                    f"additional_kwargs={res.message.additional_kwargs}\n"
                    f"raw={getattr(res, 'raw', 'N/A')}"
                )
                chunk_result["anonymized_text"] = "[EMPTY RESPONSE]"
                chunk_result["error"] = "LLM returned empty content"

        except asyncio.CancelledError:
            chunk_result["anonymized_text"] = "[CANCELLED]"
            chunk_result["error"] = "Request cancelled (timeout or interrupt)"
            raw_debug = "asyncio.CancelledError"
        except Exception as e:
            chunk_result["anonymized_text"] = f"[ERR] {type(e).__name__}: {e}"
            chunk_result["error"] = str(e)
            raw_debug = repr(e)

        # Log to trace file: original + thinking + result
        _write_trace(i, pbar_total, chunk, thinking_text, chunk_result["anonymized_text"], raw_debug)

        results.append(chunk_result)

        progress = f"[{i + 1}/{pbar_total}]"
        status = "OK" if not chunk_result["error"] else f"ERR: {chunk_result['error']}"
        print(f"  {progress} channel={chunk['channel']} date={chunk['date']} — {status}")

    return results


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------

def _write_trace(
    idx: int,
    total: int,
    chunk: dict,
    thinking: str,
    result_text: str,
    raw_debug: str = "",
) -> None:
    """Append a trace entry for one chunk to the log file."""
    os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)

    header = (
        f"{'=' * 60}\n"
        f"Chunk {idx + 1}/{total} | channel={chunk['channel']} | "
        f"date={chunk['date']} | chunk_id={chunk['chunk_id']}\n"
        f"{'=' * 60}"
    )

    sections = [
        header,
        "\n--- ORIGINAL TEXT ---",
        chunk["text"],
        "\n--- THINKING ---",
        thinking if thinking else "(no thinking output)",
    ]

    if raw_debug:
        sections.extend(["\n--- RAW DEBUG ---", raw_debug])

    sections.extend([
        "\n--- RESULT TEXT ---",
        result_text,
        "\n",
    ])

    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(sections) + "\n")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(results: List[dict], total_available: int, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    output = {
        "metadata": {
            "generated_at":        now,
            "model":               "qwen3.5:9b",
            "sample_size":          len(results),
            "seed":                 RANDOM_SEED,
            "total_chunks_available": total_available,
            "words_per_chunk":      WORDS_PER_CHUNK,
            "min_token_bypass":     MIN_TOKEN_BYPASS,
        },
        "chunks": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

    # Clear previous trace log
    if os.path.exists(TRACE_LOG_PATH):
        os.remove(TRACE_LOG_PATH)
        print(f"Cleared previous trace log: {TRACE_LOG_PATH}")

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

    # Step 2: Sample
    random.seed(RANDOM_SEED)
    sample = random.sample(chunks, min(SAMPLE_COUNT, len(chunks)))
    print(f"Sampled {len(sample)} chunks (seed={RANDOM_SEED})")

    # Step 3: Build name map
    name_map = build_name_map(sample)
    print(f"Unique names in sample: {len(name_map)}")
    for real, fake in sorted(name_map.items()):
        print(f"  {real} → {fake}")
    print()

    # Step 4: Anonymize via LLM
    print("Starting anonymization with qwen3.5:9b...")
    results = await anonymise_chunks(sample, name_map)

    # Step 5: Write output
    write_json(results, total_available=len(chunks), output_path=OUTPUT_JSON)


if __name__ == "__main__":
    asyncio.run(main())