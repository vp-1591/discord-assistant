import os
import json
import re
from typing import List
from collections import defaultdict
from tqdm import tqdm
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from src.config.config import PERSIST_DIR, MESSAGES_DIR
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

def get_unprocessed_nodes(id_map: dict) -> List[TextNode]:
    """Reads messages_json, checks against processed_messages cache, and yields only new TextNodes."""
    processed_cache_path = os.path.join("cache", "processed_messages.json")
    processed_msg_ids = set()
    
    if os.path.exists(processed_cache_path):
        try:
            with open(processed_cache_path, "r", encoding="utf-8") as f:
                processed_msg_ids = set(json.load(f))
        except Exception as e:
            sys_logger.warning(f"Could not load processed_messages cache: {e}")

    if not os.path.exists(MESSAGES_DIR):
        return [], processed_msg_ids

    new_data = []
    link_pattern = re.compile(r'^https?://[^\s]+$')
    code_pattern = re.compile(r'^[A-Za-z]{3}-[A-Za-z]{4}$')
    
    newly_processed_ids = set()

    json_files = [f for f in os.listdir(MESSAGES_DIR) if f.endswith(".json")]
    
    for filename in tqdm(json_files, desc="Reading JSON history"):
        path = os.path.join(MESSAGES_DIR, filename)
        
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list): continue
                
                last_user, last_msg = None, None
                for i, msg in enumerate(data):
                    if i % 10000 == 0 and i > 0:
                        sys_logger.info(f"Scanning {filename}: Processed {i} messages...")
                        
                    msg_id = msg.get("message_id")
                    if not msg_id or msg_id in processed_msg_ids:
                        continue
                        
                    raw_text = str(msg.get("message", "")).strip()
                    user_id = str(msg.get("user_id", "Unknown"))
                    fallback_names = msg.get("last_known_names", {})
                    
                    text = resolve_all_mentions(raw_text, id_map, fallback_names)
                    author_name = id_map.get(user_id, fallback_names.get(user_id, "Unknown"))
                    
                    if link_pattern.search(text) or code_pattern.search(text): 
                        newly_processed_ids.add(msg_id)
                        continue
                        
                    if author_name == last_user and text == last_msg: 
                        newly_processed_ids.add(msg_id)
                        continue
                        
                    last_user, last_msg = author_name, text
                    ts = msg.get("timestamp", "2000-01-01T00:00:00")
                    date_str = ts.split("T")[0]
                    channel = msg.get("channel", "unknown")
                    
                    new_data.append({
                        "id": msg_id, "user": author_name, "text": text, 
                        "date": date_str, "channel": channel
                    })
                    newly_processed_ids.add(msg_id)
            except Exception as e:
                sys_logger.error(f"Error reading {filename}: {e}")

    if not new_data:
        return [], processed_msg_ids
        
    groups = defaultdict(list)
    for entry in new_data:
        groups[(entry['channel'], entry['date'])].append(entry)

    nodes = []
    CHUNK_LIMIT = 600
    OVERLAP = 60

    def split_long_message(author: str, text: str, metadata: dict, base_id: str):
        remaining_text = text
        is_first = True
        part_idx = 0
        while remaining_text:
            prefix = f"{author}: " if is_first else f"{author}: [CUT] "
            max_chunk_len = CHUNK_LIMIT - len(prefix) - 10 
            
            node_id = f"{base_id}_p{part_idx}"
            
            if len(remaining_text) <= max_chunk_len:
                nodes.append(TextNode(text=prefix + remaining_text, metadata=metadata, id_=node_id))
                break
                
            split_idx = remaining_text.rfind(" ", 0, max_chunk_len)
            if split_idx <= 0: split_idx = max_chunk_len
            
            nodes.append(TextNode(text=prefix + remaining_text[:split_idx] + " [CUT]", metadata=metadata, id_=node_id))
            
            overlap_anchor = split_idx - OVERLAP
            new_start_idx = split_idx if overlap_anchor <= 0 else remaining_text.rfind(" ", 0, overlap_anchor)
            if new_start_idx == -1 or new_start_idx >= split_idx: new_start_idx = split_idx
            
            remaining_text = remaining_text[new_start_idx:].strip()
            is_first = False
            part_idx += 1

    for (channel, date), messages in groups.items():
        metadata = {"date": date, "channel": channel}
        current_accumulated = ""
        current_ids = []
        
        for m in messages:
            line = f"{m['user']}: {m['text']}\n"
            if len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    combo_id = "_".join(current_ids[:3]) + f"_grp{len(current_ids)}"
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata, id_=combo_id))
                    current_accumulated = ""
                    current_ids = []
                split_long_message(m['user'], m['text'], metadata, base_id=m['id'])
                continue
                
            if len(current_accumulated) + len(line) > CHUNK_LIMIT:
                if current_accumulated:
                    combo_id = "_".join(current_ids[:3]) + f"_grp{len(current_ids)}"
                    nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata, id_=combo_id))
                current_accumulated = line
                current_ids = [m['id']]
            else:
                current_accumulated += line
                current_ids.append(m['id'])
                
        if current_accumulated:
            combo_id = "_".join(current_ids[:3]) + f"_grp{len(current_ids)}"
            nodes.append(TextNode(text=current_accumulated.strip(), metadata=metadata, id_=combo_id))
            
    # Combine old and new processed IDs to return state
    updated_processed_ids = processed_msg_ids.union(newly_processed_ids)
    return nodes, updated_processed_ids

def load_or_build_index(id_map: dict):
    os.makedirs("cache", exist_ok=True)
    
    if not os.path.exists(PERSIST_DIR):
        sys_logger.info("No existing index found. Building from scratch...")
        nodes, processed_ids = get_unprocessed_nodes(id_map)
        if not nodes:
            nodes = [TextNode(text="Добряк тут!", metadata={"date": "2023-12-23"}, id_="genesis_node")]
        
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
            
            with open(os.path.join("cache", "processed_messages.json"), "w") as f:
                json.dump(list(processed_ids), f)
                
        finally:
            from llama_index.core import Settings
            index._embed_model = Settings.embed_model
            sys_logger.info("Embedding model keep_alive reset to 0 (VRAM freed).")
    else:
        sys_logger.info("Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        nodes = list(index.docstore.docs.values())
        
    return index, nodes

def insert_new_nodes(index, id_map: dict) -> List[TextNode]:
    """Finds unindexed messages, creates nodes, inserts them, and saves the DB."""
    sys_logger.info("Checking for new messages to index...")
    new_nodes, updated_processed_ids = get_unprocessed_nodes(id_map)
    
    if not new_nodes:
        sys_logger.info("No new messages to inject into the index.")
        return []
        
    print(f"🚀 Found {len(new_nodes)} new nodes. Starting Ollama embedding...")
    sys_logger.info(f"Found {len(new_nodes)} new chunked nodes to inject. Temporarily spinning up embedder...")
    
    for node in tqdm(new_nodes, desc="Preparing nodes"):
        node.excluded_llm_metadata_keys = []
        node.metadata_template = "{key}: {value}"
        node.text_template = "Metadata: {metadata_str}\nContent: {content}"
        
    build_embed_model = OllamaEmbedding(
        model_name="bge-m3", 
        base_url="http://localhost:11434",
        request_timeout=600.0,
        keep_alive="30s",
    )
    
    old_embed_model = index._embed_model
    index._embed_model = build_embed_model
    
    try:
        sys_logger.info("Ollama is now embedding nodes (this may take a while)...")
        index.insert_nodes(new_nodes)
        sys_logger.info("Nodes embedded. Persisting storage context...")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        
        with open(os.path.join("cache", "processed_messages.json"), "w") as f:
            json.dump(list(updated_processed_ids), f)
            
        print(f"✅ Successfully added {len(new_nodes)} nodes.")
        sys_logger.info(f"Successfully inserted and saved {len(new_nodes)} nodes into vector DB.")
    finally:
        index._embed_model = old_embed_model
        sys_logger.info("Injection complete. Embedder spinning down.")
        
    return new_nodes
