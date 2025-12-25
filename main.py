import discord
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from export_chat import export_chat_to_json, resolve_mentions

load_dotenv()

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents = discord.Intents.all())
        self.synced = False 

self_id = 1208704829665447947
client = aclient()

@client.event
async def on_ready():
    print("Connecting...")
    await client.wait_until_ready()
    print("Connected")

@client.event
async def on_message(message):
    if message.author.id == self_id: return 
    channel = message.channel
    # if channel.id == 863784357663211540: return  # system channel
    
    if message.content.startswith("!test"):
        async for message in channel.history(limit=10):
            msg = resolve_mentions(message)
            print(msg)
        return

    # if admin(Itadara) writes !export
    if message.content.startswith("!export") and str(message.author.id) == "470892009440149506": 
        print("Exporting chat...")
        await export_chat_to_json(channel)
        return


client.run(os.getenv('TOKEN'))