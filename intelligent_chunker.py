import json
import re
from datetime import datetime, timedelta
from typing import List, Dict

def is_spam(text: str) -> bool:
    """
    Aggressive spam filter.
    0. Empty/Too Short: < 2 chars.
    1. URL/Invite Check: http, https, discord.gg.
    2. Long Word Check: > 40 chars.
    3. Repetition Check: Sequence repeated > 3 times (e.g. kkkk).
    """
    # 0. Empty/Too Short
    if len(text.strip()) < 2:
        return True

    # 1. URL/Invite Check
    if "http" in text or "https" in text or "discord.gg" in text:
        return True
    
    # 2. Long Word Check
    words = text.split()
    if any(len(w) > 40 for w in words):
        return True
        
    # 3. Repetition Check
    # Pattern: Capturing group (.+?) repeated 3 or more times immediately matches
    # This covers "aaaa" (one char repeated 3 more times) and "hahahaha"
    if re.search(r'(.+?)\1{3,}', text):
        return True
        
    return False

def intelligent_time_based_chunking(messages: List[Dict]) -> List[str]:
    """
    Groups messages into sessions based on time gaps and character limits.
    Includes context overlap from previous chunks.
    """
    if not messages:
        return []

    # 1. Parse and Sanitize
    processed_msgs = []
    for m in messages:
        # Basic field extraction
        raw_text = str(m.get("message", m.get("content", ""))).strip()
        
        # --- SPAM FILTER ---
        if is_spam(raw_text):
            continue

        user = m.get("user", m.get("author", "Unknown"))
        ts_str = m.get("timestamp", "")

        # Handle Edge Case: Truncate huge messages
        if len(raw_text) > 2000:
            text = raw_text[:1989] + " [TRUNCATED]"
        else:
            text = raw_text.replace("\n", " ")

        # Parse Timestamp
        try:
            # Handle both ISO and custom Discord formats
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            dt = datetime.min

        processed_msgs.append({
            "dt": dt,
            "display_ts": dt.strftime("%Y-%m-%d %H:%M") if dt != datetime.min else "0000-00-00 00:00",
            "user": user,
            "text": text
        })

    # 2. Sort by time to ensure session logic works
    processed_msgs.sort(key=lambda x: x["dt"])

    final_chunks = []
    current_session = []  # List of (is_overlap_flag, msg_dict)
    current_len = 0
    
    CHUNK_LIMIT = 2000
    GAP_LIMIT = timedelta(minutes=60)
    OVERLAP_SIZE = 3

    for i, msg in enumerate(processed_msgs):
        # Prepare the formatted lines
        msg_line = f"[{msg['display_ts']}] {msg['user']}: {msg['text']}"
        msg_line_context = f"[PREVIOUS CONTEXT]: {msg_line}"
        
        # Decide if we need to split
        split_needed = False
        
        if i > 0:
            # Condition A: Time Gap > 60m
            if msg["dt"] != datetime.min and processed_msgs[i-1]["dt"] != datetime.min:
                if msg["dt"] - processed_msgs[i-1]["dt"] > GAP_LIMIT:
                    split_needed = True
            
        # Condition B: Size Limit
        # We check if adding THIS message exceeds 2000 chars
        if current_len + len(msg_line) + 1 > CHUNK_LIMIT:
            split_needed = True

        if split_needed and current_session:
            # --- Flush Current Chunk ---
            chunk_text = "\n".join([
                (f"[PREVIOUS CONTEXT]: [{m['display_ts']}] {m['user']}: {m['text']}" if is_ov else f"[{m['display_ts']}] {m['user']}: {m['text']}")
                for is_ov, m in current_session
            ])
            final_chunks.append(chunk_text)

            # --- Start New Chunk with Overlap ---
            # Extract last 3 actual messages (ignoring whether they were overlap themselves)
            last_msgs = [m for is_ov, m in current_session][-OVERLAP_SIZE:]
            
            current_session = [(True, m) for m in last_msgs]
            current_len = sum(len(f"[PREVIOUS CONTEXT]: [{m['display_ts']}] {m['user']}: {m['text']}") + 1 for m in last_msgs)
            
            # Check if current msg is now small enough for the new chunk context
            # (If it's still too big, it will just trigger another split immediately)
            current_session.append((False, msg))
            current_len += len(msg_line) + 1
        else:
            # Add to current chunk
            current_session.append((False, msg))
            current_len += len(msg_line) + 1

    # Final wrap-up
    if current_session:
        chunk_text = "\n".join([
            (f"[PREVIOUS CONTEXT]: [{m['display_ts']}] {m['user']}: {m['text']}" if is_ov else f"[{m['display_ts']}] {m['user']}: {m['text']}")
            for is_ov, m in current_session
        ])
        final_chunks.append(chunk_text)

    return final_chunks

# --- Example usage/test ---
if __name__ == "__main__":
    example_msgs = [
        {"timestamp": "2023-10-01T10:00:00", "user": "Alice", "message": "Hey everyone!"},
        {"timestamp": "2023-10-01T10:05:00", "user": "Bob", "message": "Hi Alice."},
        {"timestamp": "2023-10-01T10:10:00", "user": "Charlie", "message": "Ready for the raid?"},
        {"timestamp": "2023-10-01T12:00:00", "user": "Alice", "message": "Sorry, gap of 2 hours happened."},
        {"timestamp": "2023-10-01T12:01:00", "user": "Bob", "message": "No problem."},
    ]
    
    chunks = intelligent_time_based_chunking(example_msgs)
    for idx, c in enumerate(chunks):
        print(f"--- CHUNK {idx+1} ---\n{c}\n")
