import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List

import re

# LangChain Imports
from langchain_core.documents import Document
import create_store_rag
from intelligent_chunker import is_spam

def load_and_group_logs(directory: str) -> List[Document]:
    print(f"Reading logs from {directory}...")
    all_messages = []
    if not os.path.exists(directory):
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_messages.extend(data)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    print(f"Found {len(all_messages)} total messages. Grouping by date...")

    # 2. Group by Channel
    grouped_data = defaultdict(list)
    for msg in all_messages:
        text = str(msg.get("message", ""))
        
        # --- AGGRESSIVE SPAM FILTER ---
        if is_spam(text):
            continue

        channel = msg.get("channel", "unknown")
        grouped_data[channel].append(msg)

    final_splits = []
    CHUNK_LIMIT = 2000
    GAP_LIMIT = timedelta(minutes=60)
    OVERLAP_SIZE = 3

    print(f"Refining messages into sessions...")
    
    for channel, channel_messages in grouped_data.items():
        # Sort by actual timestamp objects for session logic
        for m in channel_messages:
            try:
                m['_dt'] = datetime.fromisoformat(m.get("timestamp", "").replace("Z", "+00:00"))
            except:
                m['_dt'] = datetime.min
        
        channel_messages.sort(key=lambda x: x['_dt'])

        current_session = [] # List of (is_overlap, msg_formatted)
        current_len = 0

        for i, m in enumerate(channel_messages):
            raw_text = str(m.get("message", "")).strip().replace("\n", " ")
            
            # Requirement 6: Truncate huge messages
            if len(raw_text) > 2000:
                raw_text = raw_text[:1980] + " [TRUNCATED]"
            
            ts_display = m['_dt'].strftime("%Y-%m-%d %H:%M") if m['_dt'] != datetime.min else "0000-00-00 00:00"
            user = m.get("user", "Unknown")
            
            msg_line = f"[{ts_display}] {user}: {raw_text}"
            
            # Extract date for metadata (from the first message of the session)
            if not current_session:
                session_date = m['_dt'].strftime("%Y-%m-%d") if m['_dt'] != datetime.min else "unknown"

            # Decision: Should we split?
            should_split = False
            if i > 0:
                # Time Gap check
                if m['_dt'] - channel_messages[i-1]['_dt'] > GAP_LIMIT:
                    should_split = True
            
            # Size Limit check
            if current_len + len(msg_line) + 1 > CHUNK_LIMIT:
                should_split = True

            if should_split and current_session:
                # Flush current
                content = "\n".join([line for _, line in current_session])
                final_splits.append(Document(
                    page_content=f"=== CHANNEL: #{channel} | DATE: {session_date} ===\n" + content,
                    metadata={"channel": channel, "date": session_date}
                ))

                # Start new with overlap
                actual_msgs = [line for ov, line in current_session if not ov][-OVERLAP_SIZE:]
                current_session = []
                current_len = 0
                # Reset session date for the new chunk
                session_date = m['_dt'].strftime("%Y-%m-%d") if m['_dt'] != datetime.min else "unknown"
                for old_line in actual_msgs:
                    overlap_line = f"[PREVIOUS CONTEXT]: {old_line}"
                    current_session.append((True, overlap_line))
                    current_len += len(overlap_line) + 1
                
                current_session.append((False, msg_line))
                current_len += len(msg_line) + 1
            else:
                current_session.append((False, msg_line))
                current_len += len(msg_line) + 1

        # Last session flush
        if current_session:
            content = "\n".join([line for _, line in current_session])
            final_splits.append(Document(
                page_content=f"=== CHANNEL: #{channel} | DATE: {session_date} ===\n" + content,
                metadata={"channel": channel, "date": session_date}
            ))
            
    return final_splits

if __name__ == "__main__":
    # 1. OPTIONAL: Reset the collection to avoid pollution
    print("Resetting vector store collection...")
    try:
        # This deletes all existing documents in this collection
        create_store_rag.vector_store.delete_collection()
        # Re-initialize the empty collection
        create_store_rag.vector_store = create_store_rag.Chroma(
            collection_name="messages_collection",
            embedding_function=create_store_rag.embeddings,
            persist_directory="./chroma_langchain_db",
        )
    except Exception as e:
        print(f"Notice: Could not reset collection (it might be empty): {e}")

    # 2. Load and chunk
    splits = load_and_group_logs("./messages_json")
    print(f"Created {len(splits)} chunks.")

    # 3. Add to store in batches
    batch_size = 50 # Smaller batches for stability
    for i in range(0, len(splits), batch_size):
        batch = splits[i : i + batch_size]
        create_store_rag.vector_store.add_documents(batch)
        print(f"Loaded: {min(i + batch_size, len(splits))} / {len(splits)}")

    print("Success! The vector store is now clean and optimized.")