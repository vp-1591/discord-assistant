"""
ingestion.py - Overhauled ingestion pipeline.

Pipeline:
  1. Read raw JSON message files; filter / de-dup against processed cache.
  2. Group messages by (channel, date).
  3. Concatenate each group into a single text block.
  4. Split with SentenceSplitter (chunk_size = MAX_CHUNK_SIZE tokens).
  5. For each chunk:
       a. Persist every raw message row to SQLite with the chunk_id.
       b. Call Ollama asynchronously (sequential) to generate a summary.
       c. Log exact prompt + response to indexing_logger.
  6. Build a TextNode(text=summary, metadata={"source_chunk_id": chunk_id, ...})
     so the vector store embeds only the summary.
  7. Insert/persist nodes into the vector store.
"""

import os
import json
import re
import uuid
import asyncio
from typing import List, Tuple
from collections import defaultdict

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding

from src.config.config import PERSIST_DIR, MESSAGES_DIR, LLM_CONTEXT_WINDOW
from src.utils.logger_setup import sys_logger, indexing_logger
from src.data.models import SessionLocal, Message
from src.config.prompts import INGESTION_SUMMARY_PROMPT

MAX_TEST_CHUNK_SIZE = int(LLM_CONTEXT_WINDOW * 0.25)
MIN_TOKEN_BYPASS = 100

# ---------------------------------------------------------------------------
# Mention resolution
# ---------------------------------------------------------------------------

def resolve_all_mentions(text: str, live_map: dict, fallback_map: dict) -> str:
    """Resolve <@id> / <@!id> / <@&id> patterns to display names."""
    pattern = re.compile(r'<@(!|&)?(\d+)>')

    def replace_match(match):
        entity_id = match.group(2)
        if entity_id in live_map:
            return f"@{live_map[entity_id]}"
        name = fallback_map.get(entity_id)
        return f"@{name}" if name else match.group(0)

    return pattern.sub(replace_match, text)


# ---------------------------------------------------------------------------
# Step 1 & 2: read JSON files, filter processed, group by (channel, date)
# ---------------------------------------------------------------------------

def _read_and_group_messages(id_map: dict) -> Tuple[dict, set, set]:
    """
    Returns:
        groups            – dict[(channel, date)] -> list of message dicts
        processed_msg_ids – set of already-indexed message IDs (from cache)
        newly_seen_ids    – set of IDs encountered this run (to update cache with)
    """
    processed_cache_path = os.path.join("cache", "processed_messages.json")
    processed_msg_ids: set = set()

    if os.path.exists(processed_cache_path):
        try:
            with open(processed_cache_path, "r", encoding="utf-8") as f:
                processed_msg_ids = set(json.load(f))
        except Exception as e:
            sys_logger.warning(f"Could not load processed_messages cache: {e}")

    if not os.path.exists(MESSAGES_DIR):
        return {}, processed_msg_ids, set()

    _link_pattern = re.compile(r'https?://\S+')
    _inline_code_pattern = re.compile(r'\b(?=.*\d)[A-Za-z0-9]{3,6}-[A-Za-z0-9]{3,6}\b')
    _noise_pattern = re.compile(
        r'-*[a-fA-F0-9]{32,128}-*|'
        r'-*[a-fA-F0-9]{8,12}-*|'
        r'-+[a-fA-F0-9]{4,12}-+|'
        r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'
    )

    groups: dict = defaultdict(list)
    newly_seen_ids: set = set()

    json_files = [f for f in os.listdir(MESSAGES_DIR) if f.endswith(".json")]

    # Pass 1: Build a global map of the absolute newest names from all JSON files
    fallback_map: dict = {}
    for filename in tqdm(json_files, desc="🔖 Building name map"):
        path = os.path.join(MESSAGES_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for msg in data:
                    fallback_map.update(msg.get("last_known_names", {}))
        except Exception:
            pass

    for filename in tqdm(json_files, desc="📂 Reading JSON history"):
        path = os.path.join(MESSAGES_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue

            last_user, last_msg = None, None
            for i, msg in enumerate(data):
                if i % 10_000 == 0 and i > 0:
                    sys_logger.info(f"Scanning {filename}: {i} msgs scanned…")

                msg_id = msg.get("message_id")
                if not msg_id or msg_id in processed_msg_ids:
                    continue

                raw_text = str(msg.get("message", "")).strip()
                user_id = str(msg.get("user_id", "Unknown"))

                text = resolve_all_mentions(raw_text, id_map, fallback_map)
                author_name = id_map.get(user_id, fallback_map.get(user_id, "Unknown"))

                newly_seen_ids.add(msg_id)

                text = _link_pattern.sub("", text)
                text = _inline_code_pattern.sub("", text)
                text = _noise_pattern.sub("", text)
                text = text.strip()
                if not text:
                    continue
                
                if author_name == last_user and text == last_msg:
                    continue

                last_user, last_msg = author_name, text

                ts = msg.get("timestamp", "2000-01-01T00:00:00")
                date_str = ts.split("T")[0]
                channel = msg.get("channel", "unknown")

                groups[(channel, date_str)].append({
                    "id": msg_id,
                    "user": author_name,
                    "text": text,
                    "date": date_str,
                    "channel": channel,
                    "timestamp": ts,
                })

        except Exception as e:
            sys_logger.error(f"Error reading {filename}: {e}")

    return groups, processed_msg_ids, newly_seen_ids


# ---------------------------------------------------------------------------
# Step 3 & 4: concatenate groups → split with SentenceSplitter
# ---------------------------------------------------------------------------

def _build_raw_chunks(groups: dict) -> List[dict]:
    """
    Returns a list of chunk dicts:
        {
            "chunk_id":  str,           # stable UUID for this chunk
            "text":      str,           # concatenated raw text for Ollama
            "date":      str,
            "channel":   str,
            "messages":  list[dict],    # original message dicts in this chunk
        }
    """
    splitter = SentenceSplitter(
        chunk_size=MAX_TEST_CHUNK_SIZE,
        chunk_overlap=100,
    )

    chunks = []

    for (channel, date), messages in tqdm(groups.items(), desc="✂️  Chunking by date"):
        # Build full concatenated text for this (channel, date) group
        full_text = "\n".join(f"{m['user']}: {m['text']}" for m in messages)
        is_small = len(full_text.split()) < MIN_TOKEN_BYPASS

        # SentenceSplitter operates on Documents; use get_nodes_from_documents
        # or the lower-level split_text method
        text_chunks = splitter.split_text(full_text)

        for chunk_text in text_chunks:
            chunk_id = str(uuid.uuid4())

            # Figure out which messages fall in this chunk (best-effort via text search)
            # We store ALL messages of the group under every chunk so the
            # SQLite lookup always returns relevant context.
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "date": date,
                "channel": channel,
                "messages": messages,
                "bypass": is_small,
            })

    return chunks


# ---------------------------------------------------------------------------
# Step 5a: persist raw messages to SQLite
# ---------------------------------------------------------------------------

def _save_messages_to_sqlite(chunk: dict) -> None:
    """Insert each message in the chunk into the SQLite Message table."""
    db = SessionLocal()
    try:
        for m in chunk["messages"]:
            # Avoid duplicate inserts (message_id is UNIQUE)
            exists = db.query(Message).filter_by(message_id=m["id"]).first()
            if exists:
                # Update chunk_id if not yet set (message belongs to a new chunk)
                if exists.chunk_id is None:
                    exists.chunk_id = chunk["chunk_id"]
                continue

            row = Message(
                message_id=m["id"],
                chunk_id=chunk["chunk_id"],
                author=m["user"],
                content=m["text"],
                timestamp=m.get("timestamp", ""),
                channel=m["channel"],
            )
            db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        sys_logger.error(f"SQLite insert error for chunk {chunk['chunk_id']}: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Step 5b: async sequential Ollama summarisation
# ---------------------------------------------------------------------------

async def _summarise_chunks(chunks: List[dict], pbar=None) -> List[TextNode]:
    """
    Process chunks strictly one-by-one (sequential, not concurrent) to
    avoid Ollama OOM.  Uses acomplete() so the discord.py event loop stays
    responsive throughout.

    Returns a list of TextNodes ready for vector-store insertion.
    """
    nodes: List[TextNode] = []
    
    # Set low temperature for factual RAG summaries
    original_temperature = getattr(Settings.llm, "temperature", None)
    Settings.llm.temperature = 0.1

    try:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            
            if chunk.get("bypass"):
                summary = chunk["text"]
                indexing_logger.info(
                    f"[CHUNK {chunk_id}] channel={chunk['channel']} date={chunk['date']}\n"
                    f"⚡ [BYPASS ACTIVE]: This chunk was skipped by the LLM and stored as RAW text.\n"
                    f"--- RAW TEXT ---\n{summary}\n{'='*60}"
                )
            else:
                prompt = INGESTION_SUMMARY_PROMPT.format(chunk=chunk["text"])

                # Log the exact prompt
                indexing_logger.info(
                    f"[CHUNK {chunk_id}] channel={chunk['channel']} date={chunk['date']}\n"
                    f"--- PROMPT ---\n{prompt}\n"
                )

                try:
                    response = await Settings.llm.acomplete(prompt)
                    summary = str(response).strip()
                except Exception as e:
                    sys_logger.error(f"Ollama error on chunk {chunk_id}: {e}")
                    # Fall back to storing the raw chunk so we don't lose data
                    summary = chunk["text"]

                # Log the exact response
                indexing_logger.info(f"[CHUNK {chunk_id}] --- SUMMARY ---\n{summary}\n{'='*60}")

            # Persist raw messages to SQLite (with chunk_id as the FK-equivalent)
            _save_messages_to_sqlite(chunk)

            # Build the vector-store node: embed the summary, link back via chunk_id
            node_type = "raw_log" if chunk.get("bypass") else "summary"
            node = TextNode(
                text=summary,
                id_=chunk_id,
                metadata={
                    "source_chunk_id": chunk_id,
                    "date": chunk["date"],
                    "channel": chunk["channel"],
                    "node_type": node_type,
                },
            )
            # Embedding only uses the summary text + date/channel metadata.
            # node_type is excluded from embeddings — it's a retrieval-time label only.
            node.excluded_embed_metadata_keys = ["source_chunk_id", "node_type"]
            node.excluded_llm_metadata_keys = []
            node.metadata_template = "{key}: {value}"
            node.text_template = "Metadata: {metadata_str}\nContent: {content}"

            nodes.append(node)
            
            if pbar is not None:
                pbar.update(1)

    finally:
        if original_temperature is not None:
            Settings.llm.temperature = original_temperature
        else:
            # We don't restore what didn't exist, but typically there's a default. Let's ensure a fallback just in case or delete the attr
            delattr(Settings.llm, "temperature")

    return nodes


# ---------------------------------------------------------------------------
# Public async helpers consumed by run_llama_index.py
# ---------------------------------------------------------------------------

async def get_raw_chunks(id_map: dict) -> Tuple[List[dict], set]:
    """
    Returns chunks and the initially loaded processed IDs.
    Does NOT summarise.
    """
    groups, processed_msg_ids, _ = _read_and_group_messages(id_map)

    if not groups:
        return [], processed_msg_ids

    sys_logger.info(f"Grouped into {len(groups)} (channel, date) buckets. Building chunks…")
    chunks = _build_raw_chunks(groups)
    return chunks, processed_msg_ids


# ---------------------------------------------------------------------------
# Index build / load / update  (called from run_llama_index.py)
# ---------------------------------------------------------------------------

def load_or_build_index(id_map: dict):
    """
    Synchronous: load existing index or build empty placeholder.
    Actual data ingestion is handled by insert_new_nodes (async).
    """
    os.makedirs("cache", exist_ok=True)

    if not os.path.exists(PERSIST_DIR):
        sys_logger.info("No existing index. Creating an empty genesis index…")
        build_embed_model = OllamaEmbedding(
            model_name="bge-m3",
            base_url="http://localhost:11434",
            request_timeout=600.0,
            keep_alive="30s",
        )
        index = VectorStoreIndex(
            [],
            show_progress=True,
            embed_model=build_embed_model,
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        from llama_index.core import Settings as _S
        index._embed_model = _S.embed_model
        sys_logger.info("Empty genesis index created and persisted.")
    else:
        sys_logger.info("Loading existing index from disk…")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    nodes = list(index.docstore.docs.values())
    return index, nodes


async def insert_new_nodes(index, id_map: dict) -> List[TextNode]:
    """
    Async: find unprocessed messages, summarise, embed, and persist.
    Called from run_llama_index.update_index().
    """
    sys_logger.info("Checking for new messages to index…")
    chunks, processed_msg_ids = await get_raw_chunks(id_map)

    if not chunks:
        sys_logger.info("No new messages to inject.")
        return []

    print(f"🚀 Found {len(chunks)} chunks to process. Starting summarisation and embedding…")
    sys_logger.info(f"Processing and Embedding {len(chunks)} chunks in batches…")

    build_embed_model = OllamaEmbedding(
        model_name="bge-m3",
        base_url="http://localhost:11434",
        request_timeout=600.0,
        keep_alive="30s",
    )

    old_embed_model = index._embed_model
    index._embed_model = build_embed_model

    all_new_nodes = []
    BATCH_SIZE = 50

    try:
        pbar = atqdm(total=len(chunks), desc="🤖 Summarising with Ollama")
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            
            # Summarise this batch
            batch_nodes = await _summarise_chunks(batch_chunks, pbar=pbar)
            if not batch_nodes:
                continue

            # Embed and insert directly to index
            await asyncio.to_thread(index.insert_nodes, batch_nodes)
            
            # Persist vector cache
            index.storage_context.persist(persist_dir=PERSIST_DIR)

            # Persist processed_messages JSON cache
            batch_msg_ids = set()
            for c in batch_chunks:
                for m in c["messages"]:
                    batch_msg_ids.add(m["id"])
            
            processed_msg_ids.update(batch_msg_ids)
            with open(os.path.join("cache", "processed_messages.json"), "w") as f:
                json.dump(list(processed_msg_ids), f)
                
            all_new_nodes.extend(batch_nodes)
            sys_logger.info(f"💾 Checkpoint saved: {len(all_new_nodes)}/{len(chunks)} chunks processed and embedded.")

        pbar.close()
        print(f"✅ Successfully indexed {len(all_new_nodes)} total summary nodes.")
        sys_logger.info(f"Successfully inserted {len(all_new_nodes)} nodes into vector DB.")
    except Exception as e:
        sys_logger.error(f"Error during node insertion: {e}")
        raise
    finally:
        index._embed_model = old_embed_model
        sys_logger.info("Embedder spinning down.")

    return all_new_nodes
