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
