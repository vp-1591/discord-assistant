import discord
import os
import json

# Configuration
OUTPUT_PREFIX = "discord_chat_part"
LINES_PER_FILE = 10000


def resolve_mentions(message, text=None):
    """
    Resolves mentions in text using the message metadata.
    If text is provided, it uses that. Otherwise, uses message.content.
    """
    content = text if text is not None else message.content
    for user in message.mentions:
        content = content.replace(f'<@{user.id}>', user.display_name).replace(f'<@!{user.id}>', user.display_name)
    for role in message.role_mentions:
        content = content.replace(f'<@&{role.id}>', role.name)
    return content

async def export_chat_to_txt(channel):
    """
    Fetches message history from a Discord channel and exports it directly to chunked .txt files.
    This merges the functionality of fetching messages and converting them to text chunks.
    
    args:
        channel: The Discord channel object (must have history attribute).
    """
    print(f"Starting export from channel: {channel.name} ({channel.id})...")
    
    # Use channel name as prefix (sanitize if necessary, but discord channel names are usually safe)
    output_prefix = channel.name
    
    file_count = 1
    line_count_current_file = 0
    total_messages_processed = 0
    
    # Open the first output file
    current_output_file_name = f"{output_prefix}_{file_count}.txt"
    current_out_f = open(current_output_file_name, 'w', encoding='utf-8')
    
    try:
        # Using oldest_first=True ensures the chat log reads chronologically (Oldest -> Newest)
        # This is generally better for RAG context flow.
        async for message in channel.history(limit=None, oldest_first=True):
            # Skip messages from bots or without content
            if message.author.bot or not message.content:
                continue
            
            text = resolve_mentions(message).strip()
            # Double check for empty text after stripping
            if not text:
                continue
                
            # Extract fields
            timestamp = message.created_at.isoformat()
            channel_name = channel.name
            author = message.author.display_name
            
            # Format: [timestamp] #channel_name <author>: text
            formatted_line = f"[{timestamp}] #{channel_name} <{author}>: {text}\n"
            
            # Check if we need to rotate files
            if line_count_current_file >= LINES_PER_FILE:
                current_out_f.close()
                print(f"Created {current_output_file_name} ({line_count_current_file} lines)")
                
                file_count += 1
                current_output_file_name = f"{output_prefix}_{file_count}.txt"
                current_out_f = open(current_output_file_name, 'w', encoding='utf-8')
                line_count_current_file = 0
            
            current_out_f.write(formatted_line)
            line_count_current_file += 1
            total_messages_processed += 1
            
            if total_messages_processed % 2000 == 0:
                print(f"Processed {total_messages_processed} messages...")
            
    except Exception as e:
        print(f"An error occurred during export: {e}")
    finally:
        current_out_f.close()
        
        # Cleanup: If the last file is empty, delete it
        if line_count_current_file == 0:
            if os.path.exists(current_output_file_name):
                os.remove(current_output_file_name)
                # Adjust final count if we removed the file
                if file_count > 1:
                    file_count -= 1
        else:
            print(f"Created {current_output_file_name} ({line_count_current_file} lines)")

    print(f"\nExport complete.")
    print(f"Total messages processed: {total_messages_processed}")
    # Calculate real file count (handling the edge case where 0 messages resulted in 0 files validly, or 1 empty file deleted)
    final_file_total = file_count if (line_count_current_file > 0 or (file_count == 1 and total_messages_processed > 0)) else (file_count - 1 if file_count > 1 else 0)
    # The simple logic above with os.remove handles the file system. 
    # Just printing the 'file_count' might be off by 1 if the last one was deleted.
    # Let's just trust the logs above for individual file creation.

async def export_chat_to_json(channel, skip_id=None):
    """
    Fetches message history from a Discord channel and exports it to a JSON file.
    
    args:
        channel: The Discord channel object (must have history attribute).
        skip_id: The ID of a message to exclude from the export.
    """
    print(f"Starting JSON export from channel: {channel.name} ({channel.id})...")
    
    messages_data = []
    
    try:
        # Using oldest_first=True ensures chronological order
        async for message in channel.history(limit=None, oldest_first=True):
            # Skip the trigger message if skip_id is provided
            if skip_id and message.id == skip_id:
                continue
                
            # Skip messages from bots or without content
            if message.author.bot or not message.content:
                continue
            
            # Extract fields and format timestamp
            timestamp = message.created_at.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Create a lookup for this specific message's entities
            known_in_msg = {str(m.id): m.display_name for m in message.mentions}
            for r in message.role_mentions:
                known_in_msg[str(r.id)] = r.name
            known_in_msg[str(message.author.id)] = message.author.display_name

            messages_data.append({
                "timestamp": timestamp,
                "channel": channel.name,
                "user_id": str(message.author.id),
                "message": message.content,
                "last_known_names": known_in_msg
            })
            
            if len(messages_data) % 2000 == 0:
                 print(f"Processed {len(messages_data)} messages...")

        os.makedirs("messages_json", exist_ok=True)
        output_filename = os.path.join("messages_json", f"{channel.name}.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)
            
        print(f"\nExport complete.")
        print(f"Total messages exported: {len(messages_data)}")
        print(f"Saved to {output_filename}")
        
    except Exception as e:
        print(f"An error occurred during JSON export: {e}")

# Usage Example (This logic would typically reside in your main bot file or a separate runner):
# if __name__ == "__main__":
#     client = discord.Client(intents=discord.Intents.all())
#     TOKEN = "YOUR_TOKEN_HERE"
#     CHANNEL_ID = 123456789
#
#     @client.event
#     async def on_ready():
#         channel = client.get_channel(CHANNEL_ID)
#         if channel:
#             await export_chat_to_txt(channel)
#         else:
#             print("Channel not found.")
#         await client.close()
#
#     client.run(TOKEN)
