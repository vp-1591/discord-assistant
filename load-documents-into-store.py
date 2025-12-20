import json
import os
from collections import defaultdict
from typing import List
from datetime import datetime

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import create_store_rag

def load_and_group_logs(directory: str) -> List[Document]:
    grouped_docs = []
    
    # 1. Read all messages into a single list first
    all_messages = []
    if not os.path.exists(directory):
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_messages.extend(data)

    # 2. Group by Date and Channel
    # key = (channel_name, date_string)
    # value = list of message objects
    grouped_data = defaultdict(list)

    for msg in all_messages:
        # Assuming timestamp format "2025-12-09T18:26:17..."
        # We slice [:10] to get "2025-12-09"
        ts = msg.get("timestamp", "")
        date_str = ts[:10] 
        channel = msg.get("channel", "unknown")
        
        key = (channel, date_str)
        grouped_data[key].append(msg)

    # 3. Convert Groups into Documents
    for (channel, date_str), messages in grouped_data.items():
        # Sort messages by time to ensure chronological order in the text
        messages.sort(key=lambda x: x.get("timestamp", ""))
        
        # A. CONSTRUCT PAGE CONTENT (The "Baking" Step)
        # We build one giant string for the whole day.
        # This solves the issue of the LLM not knowing who said what.
        content_lines = []
        unique_users = set()
        unique_mentions = set()
        unique_roles = set()
        
        for m in messages:
            short_time = m.get("timestamp", "")[11:16] # Extract "18:26"
            user = m.get("user", "Unknown")
            text = m.get("message", "")
            
            # This line preserves the "metadata" for the LLM's eyes
            line = f"[{short_time}] <{user}>: {text}"
            content_lines.append(line)
            
            # Collect user for the Vector Store metadata
            unique_users.add(user)
            unique_mentions.update(m.get("mentions", []))
            unique_roles.update(m.get("roles", []))

        full_page_content = "\n".join(content_lines)
        
        # B. CONSTRUCT METADATA (The "Filtering" Step)
        # We store the list of all users present in this conversation
        # Note: ChromaDB handles list-based metadata, but some vector stores might need strings.
        # It's safer to join them as a string for broad compatibility.
        doc_metadata = {
            "source": f"logs_{channel}_{date_str}",
            "channel": channel,
            "date": date_str,
            "users": list(unique_users), 
            "mentions": list(unique_mentions),
            "roles": list(unique_roles),
            "message_count": len(messages)
        }
        
        doc = Document(page_content=full_page_content, metadata=doc_metadata)
        grouped_docs.append(doc)
        
    return grouped_docs

# --- USAGE ---

# 1. Load grouped documents
docs = load_and_group_logs("./messages_json")
print(f"Created {len(docs)} daily summaries.")

# 2. Split (Crucial Step)
# Even though we grouped them, a single day might be HUGE (10k messages).
# We still need to split, but now we split a coherent conversation block 
# rather than tiny isolated sentences.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Larger chunks to keep conversation flow
    chunk_overlap=100,
    separators=["\n"] # Split cleanly at line breaks (messages)
)

final_splits = text_splitter.split_documents(docs)

# Note: text_splitter automatically copies the 'metadata' (users, date) 
# from the parent Document to each new split chunk!
print(f"Final chunks for vector store: {len(final_splits)}")

# Example of what a chunk looks like:
# Content: 
#   [18:26] <August>: куда лезеш
#   [18:26] <Barmacar>: ты о чем
#   ... (2000 chars of chat) ...
# Metadata: {'date': '2025-12-09', 'users': 'August, Barmacar', ...}

#Load documents into vector store
document_ids = create_store_rag.vector_store.add_documents(documents=final_splits)

print("Finished loading documents into vector store")
print(document_ids[:3])