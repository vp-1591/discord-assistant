import json
import os

# Configuration
INPUT_FILE = "discord_messages_765173252866703383.jsonl"
OUTPUT_PREFIX = "discord_chat_part"
LINES_PER_FILE = 10000

def convert_jsonl_to_txt_chunks():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    print(f"Starting conversion of '{INPUT_FILE}'...")
    
    file_count = 1
    line_count_current_file = 0
    total_messages_processed = 0
    
    # Open the first output file
    current_output_file_name = f"{OUTPUT_PREFIX}_{file_count}.txt"
    current_out_f = open(current_output_file_name, 'w', encoding='utf-8')
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as in_f:
            for line_no, line in enumerate(in_f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_no}")
                    continue

                # Extract fields
                timestamp = data.get("timestamp", "")
                channel_name = data.get("channel_name", "unknown")
                author = data.get("author", "unknown")
                text = data.get("text", "")

                # Clean Data: Skip empty text
                if not text or not text.strip():
                    continue

                # Format: [timestamp] #channel_name <author>: text
                # We normalize the timestamp a bit to be cleaner if needed, 
                # but raw ISO string is fine as requested.
                formatted_line = f"[{timestamp}] #{channel_name} <{author}>: {text}\n"

                # Check if we need to rotate files
                if line_count_current_file >= LINES_PER_FILE:
                    current_out_f.close()
                    print(f"Created {current_output_file_name} ({line_count_current_file} messages)")
                    
                    file_count += 1
                    current_output_file_name = f"{OUTPUT_PREFIX}_{file_count}.txt"
                    current_out_f = open(current_output_file_name, 'w', encoding='utf-8')
                    line_count_current_file = 0

                current_out_f.write(formatted_line)
                line_count_current_file += 1
                total_messages_processed += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        current_out_f.close()
        # If the last file is empty (rare edge case), delete it
        if line_count_current_file == 0 and os.path.exists(current_output_file_name):
            os.remove(current_output_file_name)
        else:
             print(f"Created {current_output_file_name} ({line_count_current_file} messages)")

    print(f"\nConversion complete.")
    print(f"Total messages processed: {total_messages_processed}")
    print(f"Total output files created: {file_count}")

if __name__ == "__main__":
    convert_jsonl_to_txt_chunks()
