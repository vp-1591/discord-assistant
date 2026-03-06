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
        self._assistant_loading = False  # guard against concurrent on_ready calls during build
        self.summaries_path = "cache/summaries.json"
        self.history_path = "cache/history.json"
        self.summaries = self._load_json(self.summaries_path)
        self.history = self._load_json(self.history_path)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_json(self, path, data):
        os.makedirs("cache", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    async def setup_assistant(self):
        if self.assistant is None and not self._assistant_loading:
            self._assistant_loading = True
            try:
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
            finally:
                self._assistant_loading = False  # always release the lock

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
            channel_id = str(channel.id)
            
            # 1. Get recent interaction history (Internal preservation)
            if channel_id not in client.history:
                # Bootstrap once from Discord if local history is empty
                print(f"📥 Bootstrapping history for channel {channel_id}...")
                bootstrapped = []
                async for msg in channel.history(limit=50):
                    if msg.id == message.id: continue
                    if msg.author.id == self_id or client.user.mentioned_in(msg):
                        author = msg.author.display_name
                        content = resolve_mentions(msg)
                        bootstrapped.insert(0, f"{author}: {content}")
                        if len(bootstrapped) >= 5: break
                client.history[channel_id] = bootstrapped
                client._save_json(client.history_path, client.history)

            previous_relevant = client.history.get(channel_id, [])
            current_summary = client.summaries.get(channel_id)
            
            # 2. Run RAG Query
            rag_response = await client.assistant.aquery(query)
            
            # 3. Agent 2: Synthesis with history, summary and persona
            final_response = await client.assistant.generate_refined_response(
                query_text=query,
                rag_response=str(rag_response),
                history=previous_relevant,
                summary=current_summary,
                agent1_prompt=getattr(rag_response, 'agent1_prompt', "")
            )

        # 4. Send response
        sent_msg = await message.reply(final_response)

        # 5. Post-Response Processing: Update History and Summary
        user_interaction = f"{message.author.display_name}: {query}"
        bot_interaction = f"{client.user.display_name}: {final_response}"
        
        # Add current transaction to internal history
        client.history[channel_id] = previous_relevant + [user_interaction, bot_interaction]
        
        if len(client.history[channel_id]) >= 6:
            # We compress the 4 oldest messages and keep the 2 newest
            to_summarize = client.history[channel_id][:4]
            client.history[channel_id] = client.history[channel_id][4:]
            
            print(f"📝 Post-response: Summarizing 4 messages for channel {channel_id}...")
            new_summary = await client.assistant.generate_summary(current_summary, to_summarize)
            client.summaries[channel_id] = new_summary
            client._save_json(client.summaries_path, client.summaries)
            print(f"✅ Summary updated for {channel_id}")
            
        client._save_json(client.history_path, client.history)
        return

    # if admin(Itadara) writes !export
    if message.content.startswith("!export") and str(message.author.id) == "470892009440149506": 
        print("Exporting chat...")
        await export_chat_to_json(channel, skip_id=message.id)
        return

client.run(os.getenv('TOKEN'))
