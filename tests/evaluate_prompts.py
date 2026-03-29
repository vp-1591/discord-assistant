"""
evaluate_prompts.py
===================
Standalone script to evaluate summarization strategies.
"""

import os
import sys
import json
import re
import random
import asyncio
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Project Setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tqdm import tqdm
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from src.config.config import configure_settings, MESSAGES_DIR, LLM_CONTEXT_WINDOW

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_COUNT      = 6
RANDOM_SEED       = 40
OUTPUT_PATH       = os.path.join(ROOT, "logs", "prompt_evaluation_results.md")

MAX_TEST_CHUNK_SIZE = int(LLM_CONTEXT_WINDOW * 0.25) # ~2048 tokens
MIN_TOKEN_BYPASS    = 100                           # Raw text if < 100 tokens

# ---------------------------------------------------------------------------
# Prompt variations
# ---------------------------------------------------------------------------

# --- v3.2 Skilled Archivist (activity-aware upgrade of v3.1) ---
_PROMPT_V32 = """СИСТЕМА: Ты — Архивариус Discord-сообщества. Твоя цель — извлечение фактов и социальной динамики для RAG-базы знаний.
ДУМАЙ ПЕРЕД ОТВЕТОМ: Определи природу чанка. Какой тип активности здесь происходит? Это влияет на то, как читать сообщения:
- Кто говорит от своего лица, а кто — от лица персонажа, нарратора, ведущего?
- Какие события здесь являются фактами (пусть даже вымышленными в рамках игры), а что — мета-комментарии?
- Есть ли смешение слоёв (игра + обсуждение вне игры, творчество + технический вопрос)?
ПРАВИЛА:
1. СТРУККТУРА: [ИМЯ (роль, если применимо): ФАКТ/СОБЫТИЕ].
2. ПРИОРИТЕТ: Технические данные, ключевые решения, значимые события — включая события вымышленного мира, если это суть активности.
3. АТРИБУЦИЯ: Пользователь и его персонаж/роль — разные сущности. Фиксируй оба уровня, если они различимы.
4. СОЦИАЛЬНЫЙ КОНТЕКСТ: Конфликты, юмор, абсурд — это социальные факты, не шум.
5. РАЗДЕЛЯЙ СЛОИ: Если в чанке смешаны типы активности, обозначь их раздельно в выводе.
6. ФОРМАТ: Без вводных слов. 3–5 предложений. Язык оригинала.
ОБРАЗЦЫ:
<пример>
Вход:
  Mira: в статье написано 2019, а не 2021
  Ozan: нет, я смотрю на таблицу 3 — там явно 2021, страница 14
Вывод:
  Ozan оспаривает Mira, указывая на таблицу 3 стр. 14:
  год публикации данных — 2021, не 2019.
</пример>

<пример>
Вход:
  Zero: "...итак, модель показала p<0.001 на выборке n=340..."
  Ozan: Zero, слайд не грузится у половины, переключи
  Zero: ой, переключила, видно теперь?
  Ozan: да, всё
  Zero: "в заключение: эффект устойчив во всех трёх когортах"
Вывод:
  [Доклад] Zero (в роли докладчика) представила результаты:
  p<0.001, n=340, нулевая гипотеза отвергнута; эффект устойчив
  в трёх когортах.
  [Орг] Слайд не отображался; Zero переключила по сигналу Ozan.
</пример>

ДАННЫЕ:
{chunk}
"""

PROMPTS = {
    "3 — Skilled Archivist (v3.2)":    _PROMPT_V32,
}

# ---------------------------------------------------------------------------
# Mention & Code Filtering (test-only formatting)
# ---------------------------------------------------------------------------

def _resolve_mentions_test(text: str, fallback_map: dict) -> str:
    """Uses [mention: Name] format for testing."""
    pattern = re.compile(r'<@(!|&)?(\d+)>')
    def replace_match(match):
        entity_id = match.group(2)
        name = fallback_map.get(entity_id)
        return f"@{name}" if name else match.group(0)
    return pattern.sub(replace_match, text)

_link_pattern        = re.compile(r'https?://\S+')
_inline_code_pattern = re.compile(r'\b(?=.*\d)[A-Za-z0-9]{3,6}-[A-Za-z0-9]{3,6}\b')
_noise_pattern = re.compile(
    r'-*[a-fA-F0-9]{32,128}-*|'           # MD5, SHA, etc.
    r'-*[a-fA-F0-9]{8,12}-*|'             # UUID fragments
    r'-+[a-fA-F0-9]{4,12}-+|'             # Padded IDs like --9207-
    r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}' # Full UUIDs
)


# ---------------------------------------------------------------------------
# Step 1: Read & group messages (Strictly by Date)
# ---------------------------------------------------------------------------

def load_and_group_messages(messages_dir: str) -> dict:
    if not os.path.exists(messages_dir):
        raise FileNotFoundError(f"messages_json directory not found: {messages_dir}")

    json_files = [f for f in os.listdir(messages_dir) if f.endswith(".json")]
    
    fallback_map: dict = {}
    for filename in tqdm(json_files, desc="🔖 Building name map"):
        path = os.path.join(messages_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for msg in data:
                fallback_map.update(msg.get("last_known_names", {}))
        except Exception: pass

    groups: dict = defaultdict(list)
    for filename in tqdm(json_files, desc="📂 Reading JSON"):
        path = os.path.join(messages_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            last_user, last_msg = None, None
            for msg in data:
                raw_text = str(msg.get("message", "")).strip()
                if not raw_text:
                    continue
                
                text = _resolve_mentions_test(raw_text, fallback_map)
                text = _link_pattern.sub("", text)
                text = _inline_code_pattern.sub("", text)
                text = _noise_pattern.sub("", text)
                
                text = text.strip()
                if not text:
                    continue
                
                user_id = str(msg.get("user_id", "Unknown"))
                author = fallback_map.get(user_id, msg.get("last_known_names", {}).get(user_id, f"User_{user_id}"))

                if author == last_user and text == last_msg: continue
                last_user, last_msg = author, text

                date = msg.get("timestamp", "2000-01-01").split("T")[0]
                channel = msg.get("channel", "unknown")
                groups[(channel, date)].append(f"{author}: {text}")
        except Exception: pass

    return groups


# ---------------------------------------------------------------------------
# Step 2: Build chunks with Token Bypass Check
# ---------------------------------------------------------------------------

def build_chunks(groups: dict) -> list:
    splitter = SentenceSplitter(
        chunk_size=MAX_TEST_CHUNK_SIZE,
        chunk_overlap=100,
    )

    chunks = []
    for (channel, date), lines in tqdm(groups.items(), desc="✂️  Chunking"):
        full_text = "\n".join(lines)
        is_small = len(full_text.split()) < MIN_TOKEN_BYPASS
        
        for chunk_text in splitter.split_text(full_text):
            chunks.append({
                "channel": channel,
                "date": date,
                "text": chunk_text,
                "bypass": is_small
            })
    return chunks


# ---------------------------------------------------------------------------
# Step 4 & 5: Sequential Ollama evaluation
# ---------------------------------------------------------------------------

async def evaluate(chunks: list, prompts: dict) -> list:
    results = []
    configure_settings()
    # Explicitly set low temperature for factual RAG summaries
    Settings.llm.temperature = 0.1
    
    pbar = tqdm(total=len(chunks), desc="🤖 Evaluating")

    for i, chunk in enumerate(chunks):
        chunk_results = {
            "index": i + 1, "channel": chunk["channel"], "date": chunk["date"], 
            "text": chunk["text"], "bypass": chunk["bypass"], "summaries": {}
        }

        if chunk["bypass"]:
            for p_name in prompts:
                chunk_results["summaries"][p_name] = "⚡ [BYPASS] Using RAW text (Too small for summarization)"
        else:
            for p_name, p_tmpl in prompts.items():
                prompt = p_tmpl.format(chunk=chunk["text"])
                try:
                    res = await Settings.llm.acomplete(prompt)
                    chunk_results["summaries"][p_name] = str(res).strip()
                except Exception as e:
                    chunk_results["summaries"][p_name] = f"ERR: {e}"
        
        results.append(chunk_results)
        pbar.update(1)
    
    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_markdown(results: list, prompts: dict, output_path: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    lines = [
        "# Prompt Evaluation Results (Bypass Mode)\n",
        f"**Generated:** {now} | **Min Tokens for Bypass:** {MIN_TOKEN_BYPASS}\n",
        f"**Chunk Size:** 0.4 context (~{MAX_TEST_CHUNK_SIZE} tokens)\n",
        "---\n"
    ]

    for r in results:
        lines.append(f"## Chunk {r['index']} — `{r['channel']}` / {r['date']}")
        if r["bypass"]:
            lines.append("> ⚡ **BYPASS ACTIVE**: This chunk was skipped by the LLM and stored as RAW text.\n")
        
        lines.append("### 📄 Source Text\n```\n" + r["text"] + "\n```\n")
        
        for p_name, summary in r["summaries"].items():
            lines.append(f"### 🔹 {p_name}\n{summary}\n")
        
        lines.append("\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅ Results: {output_path}")


async def main():
    groups = load_and_group_messages(MESSAGES_DIR)
    chunks = build_chunks(groups)
    random.seed(RANDOM_SEED)
    sample  = random.sample(chunks, SAMPLE_COUNT) if len(chunks) > SAMPLE_COUNT else chunks
    
    results = await evaluate(sample, PROMPTS)
    write_markdown(results, PROMPTS, OUTPUT_PATH)

if __name__ == "__main__":
    asyncio.run(main())
