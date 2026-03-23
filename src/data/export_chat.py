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
    Fetches message history from a Discord channel incrementally and exports it to a JSON file.
    Saves in chunks to survive interruptions.
    
    Returns:
        int: The number of new messages successfully exported.
    """
    print(f"Starting JSON export from channel: {channel.name} ({channel.id})...")
    
    os.makedirs("messages_json", exist_ok=True)
    output_filename = os.path.join("messages_json", f"{channel.name}.json")
    
    existing_messages = []
    last_message_obj = None
    
    # Check for existing export to find the last message ID
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_messages = json.load(f)
                if existing_messages:
                    last_msg_data = existing_messages[-1]
                    try:
                        last_message_obj = await channel.fetch_message(int(last_msg_data['message_id']))
                        print(f"Found existing export. Resuming after message ID: {last_message_obj.id}")
                    except discord.NotFound:
                        print("Warning: Last message in JSON was deleted from Discord. Exporting all.")
                        last_message_obj = None
        except Exception as e:
            print(f"Warning: Could not read existing export ({e}). Starting fresh.")
            existing_messages = []

    new_messages_data = []
    total_new_exported = 0
    save_chunk_size = 2000

    try:
        # Fetch only messages after the last known message
        async for message in channel.history(limit=None, oldest_first=True, after=last_message_obj):
            if skip_id and message.id == skip_id:
                continue
                
            if message.author.bot or not message.content:
                continue
            
            timestamp = message.created_at.strftime('%Y-%m-%dT%H:%M:%S')
            
            known_in_msg = {str(m.id): m.display_name for m in message.mentions}
            for r in message.role_mentions:
                known_in_msg[str(r.id)] = r.name
            known_in_msg[str(message.author.id)] = message.author.display_name

            new_messages_data.append({
                "message_id": str(message.id),
                "timestamp": timestamp,
                "channel": channel.name,
                "user_id": str(message.author.id),
                "message": message.content,
                "last_known_names": known_in_msg
            })
            
            # Save chunk to disk to survive interruptions
            if len(new_messages_data) >= save_chunk_size:
                existing_messages.extend(new_messages_data)
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(existing_messages, f, ensure_ascii=False, indent=2)
                
                total_new_exported += len(new_messages_data)
                print(f"Saved chunk of {len(new_messages_data)} messages. Total new: {total_new_exported}...")
                new_messages_data = []

        # Save remaining messages
        if new_messages_data:
            existing_messages.extend(new_messages_data)
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_messages, f, ensure_ascii=False, indent=2)
            total_new_exported += len(new_messages_data)
            
        print(f"\nExport complete.")
        print(f"New messages exported: {total_new_exported}")
        print(f"Total messages in archive: {len(existing_messages)}")
        return total_new_exported
        
    except Exception as e:
        print(f"An error occurred during JSON export (saved progress safely): {e}")
        return total_new_exported
