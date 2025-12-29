import discord
import os
from dotenv import load_dotenv
from export_chat import export_chat_to_json, resolve_mentions


from run_llama_index import RAGAssistant

load_dotenv()

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents = discord.Intents.all())
        self.synced = False 
        self.assistant = None

    async def setup_assistant(self):
        if self.assistant is None:
            print("🤖 Initializing RAG Assistant...")
            self.assistant = RAGAssistant()
            print("✅ RAG Assistant ready.")

self_id = 1208704829665447947
client = aclient()

@client.event
async def on_ready():
    print("Connecting...")
    await client.setup_assistant()
    await client.wait_until_ready()
    print(f"Connected as {client.user}")

@client.event
async def on_message(message):
    if message.author.id == self_id or message.author.bot: return 
    
    channel = message.channel

    #if channel.id != 862031269844090880: return #restrict bot only to test channel

    # Check if bot is mentioned
    if client.user.mentioned_in(message):
        # Extract query (remove mention)

        query = message.content.replace(f'<@{self_id}>', '').replace(f'<@!{self_id}>', '').strip()
        query = resolve_mentions(message, text=query)
        if not query:
            query = "Привет!" # Default query if just mention

        print(f"💬 Mentions detected. Query: {query}")
        
        async with channel.typing():
            # 1. Get recent history (e.g., last 5 messages)
            history = []
            async for msg in channel.history(limit=5):
                if msg.id == message.id: continue # Skip current message
                author = msg.author.display_name
                content = resolve_mentions(msg)
                history.insert(0, f"{author}: {content}")

            # 2. Run RAG Query
            rag_response = await client.assistant.aquery(query)
            
            # 3. Agent 2: Synthesis with history and persona
            final_response = await client.assistant.generate_refined_response(
                query_text=query,
                rag_response=str(rag_response),
                history=history
            )

            # 4. Send response
            await channel.send(final_response)
        return

    # if admin(Itadara) writes !export
    if message.content.startswith("!export") and str(message.author.id) == "470892009440149506": 
        print("Exporting chat...")
        await export_chat_to_json(channel)
        return

client.run(os.getenv('TOKEN'))
