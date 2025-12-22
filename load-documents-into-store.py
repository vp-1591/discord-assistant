import json
import os
from collections import defaultdict
from typing import List

import re

# LangChain Imports
from langchain_core.documents import Document
import create_store_rag

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

    # Group by Date and Channel
    grouped_data = defaultdict(list)
    for msg in all_messages:
        text = str(msg.get("message", ""))
        
        # --- NOISE FILTERING ---
        if len(text.strip()) < 2: continue
        
        # Skip repetitive spam (4 or more times in a row)
        words = text.split()
        if len(words) >= 4:
            is_spam = False
            consecutive_count = 1
            for i in range(1, len(words)):
                if words[i] == words[i-1]:
                    consecutive_count += 1
                    if consecutive_count >= 4:
                        is_spam = True
                        break
                else:
                    consecutive_count = 1
            if is_spam: continue

        # Skip random alphanumeric codes/invites
        if re.match(r'^[a-zA-Z0-9]{3}-[a-zA-Z0-9]{4}$', text.strip()): continue
        if "68M-L0xY" in text: continue

        # Skip messages that are ONLY a link (e.g. tenor.com, https://...)
        if re.match(r'^https?://[^\s]+$', text.strip()): continue
        if "tenor.com" in text and len(text.split()) == 1: continue

        ts = msg.get("timestamp", "")
        date_str = ts[:10] 
        channel = msg.get("channel", "unknown")
        key = (channel, date_str)
        grouped_data[key].append(msg)

    final_splits = []
    CHUNK_SIZE_LIMIT = 700 # Smaller chunks = Higher semantic weight for keywords
    
    for (channel, date_str), messages in grouped_data.items():
        messages.sort(key=lambda x: x.get("timestamp", ""))
        
        # Header for every chunk in this group
        header = f"=== CHANNEL: #{channel} | DATE: {date_str} ===\n"
        current_chunk_lines = []
        current_length = len(header)

        for m in messages:
            short_time = m.get("timestamp", "")[11:16]
            user = m.get("user", "Unknown")
            # Keep message content on one line
            text = str(m.get("message", "")).strip().replace("\n", " ")
            
            # Format: [Time] <User>: Message
            msg_line = f"[{short_time}] <{user}>: {text}"
            msg_len = len(msg_line) + 1 # +1 for newline character
            
            # If addition would exceed limit, flush current chunk
            if current_length + msg_len > CHUNK_SIZE_LIMIT and current_chunk_lines:
                content = header + "\n".join(current_chunk_lines)
                final_splits.append(Document(
                    page_content=content,
                    metadata={"channel": channel, "date": date_str}
                ))
                current_chunk_lines = []
                current_length = len(header)

            # Handle messages that are individually longer than the limit
            if msg_len > CHUNK_SIZE_LIMIT:
                # Flush existing if any
                if current_chunk_lines:
                    content = header + "\n".join(current_chunk_lines)
                    final_splits.append(Document(page_content=content, metadata={"channel": channel, "date": date_str}))
                    current_chunk_lines = []
                    current_length = len(header)
                
                # Split large message with repeated context
                prefix = f"=== CHANNEL: #{channel} | DATE: {date_str} ===\n[{short_time}] <{user}>: "
                available = CHUNK_SIZE_LIMIT - len(prefix) - 10
                parts = [text[i:i+available] for i in range(0, len(text), available)]
                for p in parts:
                    final_splits.append(Document(
                        page_content=prefix + p + " (part)",
                        metadata={"channel": channel, "date": date_str, "user": user}
                    ))
                continue

            current_chunk_lines.append(msg_line)
            current_length += msg_len

        # Final flush for the day
        if current_chunk_lines:
            content = header + "\n".join(current_chunk_lines)
            final_splits.append(Document(
                page_content=content,
                metadata={"channel": channel, "date": date_str}
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