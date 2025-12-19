import discord
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from export_chat import export_chat_to_json

load_dotenv()

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents = discord.Intents.all())
        self.synced = False 
        self.genai_client = None
        self.rag_store_name = None

self_id = 1208704829665447947
client = aclient()
STORE_DISPLAY_NAME = "My_Project_Knowledge_Base"

@client.event
async def on_ready():
    print("Connecting...")
    await client.wait_until_ready()
    print("Connected")
    
    

@client.event
async def on_message(message):
    if message.author.id == self_id: return 
    channel = message.channel
    if channel.id == 863784357663211540: return 

    if message.content.startswith("!export"):
        await export_chat_to_json(channel)
        return


client.run(os.getenv('TOKEN'))