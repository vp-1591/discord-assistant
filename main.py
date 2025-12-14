import discord
import os
from dotenv import load_dotenv
from export_messages import export_messages_for_rag
from google import genai
from google.genai import types

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
    
    # Initialize Gemini Client
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            client.genai_client = genai.Client(api_key=api_key)
            print("Gemini Client Initialized.")
            
            # Find the RAG Store
            print(f"Searching for Knowledge Base: '{STORE_DISPLAY_NAME}'...")
            existing_stores = list(client.genai_client.file_search_stores.list())
            for store in existing_stores:
                if store.display_name == STORE_DISPLAY_NAME:
                    client.rag_store_name = store.name
                    break
            
            if client.rag_store_name:
                print(f"RAG Store Found: {client.rag_store_name}")
            else:
                print(f"ERROR: RAG Store '{STORE_DISPLAY_NAME}' not found.")
                
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
    else:
        print("ERROR: GEMINI_API_KEY not found in env.")

@client.event
async def on_message(message):
    if message.author.id == self_id: return 
    channel = message.channel
    if channel.id == 863784357663211540: return 

    # Check if bot is mentioned
    if client.user.mentioned_in(message):
        if not client.genai_client or not client.rag_store_name:
            await message.reply("My memory is not connected right now.")
            return

        async with message.channel.typing():
            try:
                # Clean prompt
                prompt = message.content.replace(f'<@{client.user.id}>', '').strip()
                if not prompt:
                    prompt = "Hello" 

                print(f"Generating RAG response for: {prompt}")

                # Call Gemini with File Search
                model_name = 'gemini-2.5-flash' # Using 2.5-flash as the standard efficient model
                response = client.genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful bot on this Discord server. Your knowledge comes from the server's chat history (Context). If the user asks about the server history or past events, rely strictly on the provided Context. If the answer is not in the Context, tell the user you couldn't find it in the logs, but try to help with the query generally if possible. Speak casually and naturally like a Discord user. Do not use formal AI phrases like 'As an AI language model'.",
                        tools=[types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[client.rag_store_name]
                            )
                        )]
                    )
                )
                
                if response.text:
                    await message.reply(response.text)
                    
                    # Print grounding metadata if available for debugging
                    if response.candidates and response.candidates[0].grounding_metadata and response.candidates[0].grounding_metadata.grounding_attributions:
                         print("Grounding Attributions:", response.candidates[0].grounding_metadata.grounding_attributions)
                    else:
                         print("No RAG context used for this response")
                else:
                    await message.reply("I found nothing in my memory about that.")

            except Exception as e:
                print(f"Generation Error: {e}")
                await message.reply("I'm having trouble accessing my memory right now.")

client.run(os.getenv('TOKEN'))