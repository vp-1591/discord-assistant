import discord
import os
import json
from dotenv import load_dotenv
from export_chat import export_chat_to_json, resolve_mentions
import asyncio

from run_llama_index import RAGAssistant

load_dotenv()

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents = discord.Intents.all())
        self.synced = False 
        self.assistant = None
        self.summaries_path = "cache/summaries.json"
        self.summaries = self._load_summaries()

    def _load_summaries(self):
        if os.path.exists(self.summaries_path):
            with open(self.summaries_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_summaries(self):
        os.makedirs("cache", exist_ok=True)
        with open(self.summaries_path, "w", encoding="utf-8") as f:
            json.dump(self.summaries, f, ensure_ascii=False, indent=4)

    async def setup_assistant(self):
        if self.assistant is None:
            from run_llama_index import FORCE_REBUILD, PERSIST_DIR
            
            id_map = {}
            # Only scan the server if we are going to rebuild the index
            if FORCE_REBUILD or not os.path.exists(PERSIST_DIR):
                print("🤖 Rebuild required. Collecting latest names and roles...")
                # Map all users the bot can see
                for user in self.users:
                    id_map[str(user.id)] = user.display_name
                
                # Map all roles and guild members
                for guild in self.guilds:
                    for role in guild.roles:
                        id_map[str(role.id)] = role.name
                    for member in guild.members:
                        id_map[str(member.id)] = member.display_name
                print(f"✅ Map built with {len(id_map)} identities.")
            else:
                print("📦 Loading existing index (skipping identity scan).")

            print("🤖 Initializing RAG Assistant (This may take a while if rebuilding)...")

            # Use to_thread to prevent blocking the heartbeat during heavy indexing
            self.assistant = await asyncio.to_thread(RAGAssistant, id_map=id_map)
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
            # 1. Get recent interaction history (find up to 5 previous messages, as we will add 2 more)
            previous_relevant = []
            async for msg in channel.history(limit=50):
                if msg.id == message.id: continue
                if msg.author.id == self_id or client.user.mentioned_in(msg):
                    author = msg.author.display_name
                    content = resolve_mentions(msg)
                    previous_relevant.insert(0, f"{author}: {content}")
                    if len(previous_relevant) >= 5: break # Only need 5 to see if we hit 6+ later

            channel_id = str(channel.id)
            current_summary = client.summaries.get(channel_id)
            
            # 2. Run RAG Query
            rag_response = await client.assistant.aquery(query)
            
            # 3. Agent 2: Synthesis with history, summary and persona
            final_response = await client.assistant.generate_refined_response(
                query_text=query,
                rag_response=str(rag_response),
                history=previous_relevant,
                summary=current_summary
            )

        # 4. Send response
        sent_msg = await message.reply(final_response)

        # 5. Post-Response Processing: Update Summary if needed
        # Combine history with the transaction that just finished
        user_interaction = f"{message.author.display_name}: {query}"
        bot_interaction = f"{client.user.display_name}: {final_response}"
        
        all_relevant = previous_relevant + [user_interaction, bot_interaction]
        
        if len(all_relevant) >= 6:
            # We compress the 4 oldest messages and keep the 2 newest
            to_summarize = all_relevant[:4]
            # No need to explicitly 'keep' the 2 newest in a variable here, 
            # because on the NEXT query, the history fetch will find them in history(limit=50)
            
            print(f"📝 Post-response: Summarizing 4 messages for channel {channel_id}...")
            new_summary = await client.assistant.generate_summary(current_summary, to_summarize)
            client.summaries[channel_id] = new_summary
            client._save_summaries()
            print(f"✅ Summary updated for {channel_id}")
        return

    # if admin(Itadara) writes !export
    if message.content.startswith("!export") and str(message.author.id) == "470892009440149506": 
        print("Exporting chat...")
        await export_chat_to_json(channel, skip_id=message.id)
        return

client.run(os.getenv('TOKEN'))
