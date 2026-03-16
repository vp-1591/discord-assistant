import os
import json
import re
from typing import List
from collections import defaultdict
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from src.config.config import PERSIST_DIR, MESSAGES_DIR, FORCE_REBUILD
from src.utils.logger_setup import sys_logger

def resolve_all_mentions(text: str, live_map: dict, fallback_map: dict) -> str:
    """
    Priority:
    1. live_map (IDs currently in Discord)
    2. fallback_map (IDs saved during export)
    3. original match string (if nothing found)
    """
    pattern = re.compile(r'<@(!|&)?(\d+)>')
    
    def replace_match(match):
        entity_id = match.group(2)
        # 1. Try live names first
        if entity_id in live_map:
            return live_map[entity_id]
        # 2. Try names that were saved when message was exported
        return fallback_map.get(entity_id, match.group(0))

    return pattern.sub(replace_match, text)

def load_nodes_from_json(directory: str, id_map: dict) -> List[TextNode]:
    if not os.path.exists(directory):
        return []

    processed_data = []
    link_pattern = re.compile(r'^https?://[^\s]+$')
    code_pattern = re.compile(r'^[A-Za-z]{3}-[A-Za-z]{4}$')

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list): continue
                    last_user, last_msg = None, None
                    for msg in data:
                        raw_text = str(msg.get("message", "")).strip()
                        user_id = str(msg.get("user_id", "Unknown"))
                        fallback_names = msg.get("last_known_names", {})
                        
                        # Resolve mentions: Live Map -> Fallback Map -> Raw ID
                        text = resolve_all_mentions(raw_text, id_map, fallback_names)
                        
                        # Resolve the author name: Live Map -> Fallback Map -> "Unknown"
                        author_name = id_map.get(user_id, fallback_names.get(user_id, "Unknown"))
                        
                        if link_pattern.search(text) or code_pattern.search(text): continue
                        if author_name == last_user and text == last_msg: continue
                        last_user, last_msg = author_name, text
                        ts = msg.get("timestamp", "2000-01-01T00:00:00")
                        date_str = ts.split("T")[0]
                        channel = msg.get("channel", "unknown")
                        processed_data.append({"user": author_name, "text": text, "date": date_str, "channel": channel})
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    groups = defaultdict(list)
    for entry in processed_data:
        groups[(entry['channel'], entry['date'])].append(entry)

    nodes = []
    CHUNK_LIMIT = 600
    OVERLAP = 60

    def split_long_message(author: str, text: str, metadata: dict):
        remaining_text = text
        is_first = True
        while remaining_text:
            prefix = f"{author}: " if is_first else f"{author}: [CUT] "
            max_chunk_len = CHUNK_LIMIT - len(prefix) - 10 
            if len(remaining_text) <= max_chunk_len:
                nodes.append(TextNode(text=prefix + remaining_text, metadata=metadata))
                break
            split_idx = remaining_text.rfind(" ", 0, max_chunk_len)
            if split_idx <= 0: split_idx = max_chunk_len
            nodes.append(TextNode(text=prefix + remaining_text[:split_idx] + " [CUT]", metadata=metadata))
            overlap_anchor = split_idx - OVERLAP
            new_start_idx = split_idx if overlap_anchor <= 0 else remaining_text.rfind(" ", 0, overlap_anchor)
            if new_start_idx == -1 or new_start_idx >= split_idx: new_start_idx = split_idx
            remaining_text = remaining_text[new_start_idx:].strip()
            is_first = False

    for (channel, date), messages in groups.items():
        metadata = {"date": date, "channel": channel}
        current_accumulated = ""
        for m in messages:
            line = f"{m['user']}: {m['text']}\n"
            if len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
                    current_accumulated = ""
                split_long_message(m['user'], m['text'], metadata)
                continue
            if len(current_accumulated) + len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
                current_accumulated = line
            else:
                current_accumulated += line
        if current_accumulated:
            nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata))
    return nodes

def load_or_build_index(id_map: dict):
    if not os.path.exists(PERSIST_DIR) or FORCE_REBUILD:
        nodes = load_nodes_from_json(MESSAGES_DIR, id_map)
        if not nodes:
            nodes = [TextNode(text="Добряк тут!", metadata={"date": "2023-12-23"})]
        
        for node in nodes:
            node.excluded_llm_metadata_keys = []
            node.metadata_template = "{key}: {value}"
            node.text_template = "Metadata: {metadata_str}\nContent: {content}"
        
        sys_logger.info(f"Building index from {len(nodes)} nodes...")
        build_embed_model = OllamaEmbedding(
            model_name="bge-m3", 
            base_url="http://localhost:11434",
            request_timeout=600.0,
            keep_alive="30s",
        )
        try:
            index = VectorStoreIndex(nodes, show_progress=True, insert_batch_size=len(nodes), embed_model=build_embed_model)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        finally:
            from llama_index.core import Settings
            index._embed_model = Settings.embed_model
            sys_logger.info("Embedding model keep_alive reset to 0 (VRAM freed).")
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        nodes = list(index.docstore.docs.values())
    return index, nodes
