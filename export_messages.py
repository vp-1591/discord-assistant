import discord
import json
from datetime import datetime

# You will need to set up your Discord client and channel here
# channel = client.get_channel(CHANNEL_ID) 

async def export_messages_for_rag(channel):
    file_path = f"discord_messages_{channel.id}.jsonl"
    count = 0
    
    with open(file_path, 'w', encoding='utf-8') as f:
        # Use async for message in channel.history(limit=None) as you planned
        async for message in channel.history(limit=None): 
            # Skip messages without content (e.g., just embeds/attachments)
            if not message.content:
                continue
            
            # 1. Create a dictionary for the message
            # The structure is clean for later parsing and metadata handling
            message_data = {
                "id": str(message.id),
                "author": message.author.display_name,
                "author_id": str(message.author.id),
                "timestamp": message.created_at.isoformat(),
                "channel_name": channel.name,
                "text": message.content
            }
            
            # 2. Write the JSON object as a single line to the file
            f.write(json.dumps(message_data) + '\n')
            count += 1
    
    print(f"Exported {count} messages to {file_path}")

# Run your client and call this function.