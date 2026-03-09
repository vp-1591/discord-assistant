import discord
import os
import json
from dotenv import load_dotenv
from export_chat import export_chat_to_json, resolve_mentions
import asyncio

from run_llama_index import RAGAssistant
from logger_setup import sys_logger, chat_logger
from opinion_manager import OpinionManager

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
        self.opinions = OpinionManager()

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

                # Get the bot's nickname from the first guild (fallback to global name)
                bot_name = self.guilds[0].me.display_name if self.guilds else self.user.display_name
                print(f"🤖 Initializing RAG Assistant as '{bot_name}' (This may take a while if rebuilding)...")

                # Use to_thread to prevent blocking the heartbeat during heavy indexing
                self.assistant = await asyncio.to_thread(RAGAssistant, id_map=id_map, name=bot_name)
                print(f"✅ RAG Assistant ready as '{bot_name}'.")
            finally:
                self._assistant_loading = False  # always release the lock

self_id = 1208704829665447947
client = aclient()

@client.event
async def on_ready():
    sys_logger.info("Connecting...")
    await client.setup_assistant()
    await client.wait_until_ready()
    sys_logger.info(f"Connected as {client.user}")

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

        sys_logger.info(f"Mentions detected. Query: {query}")
        
        async with channel.typing():
            channel_id = str(channel.id)
            
            # 1. Get recent interaction history (Internal preservation)
            if channel_id not in client.history:
                # Rely strictly on local history. Bootstrapping from Discord is removed.
                sys_logger.info(f"Initialized empty local history for channel {channel_id}.")
                client.history[channel_id] = []
                client._save_json(client.history_path, client.history)

            previous_relevant = client.history.get(channel_id, [])
            current_summary = client.summaries.get(channel_id)
            
            try:
                # 2. Run RAG Query
                rag_response = await client.assistant.aquery(query)
                
                # Opinion Management: Get User Profile
                user_name = message.author.display_name
                user_id = message.author.id
                user_profile_data = client.opinions.get_user_profile(user_id)
                user_profile_str = ""
                if user_profile_data:
                    user_profile_str = f'<user_profile name="{user_name}">\nStance: {user_profile_data.get("head_of_archive_stance")}\nHistory: {user_profile_data.get("interaction_history")}\n</user_profile>'
                
                # Opinion Management: Find Targets via Fuzzy Match
                targets = client.opinions.find_targets(query)
                target_profiles_str = ""
                for t_id, t_data in targets:
                    if str(t_id) != str(user_id):  # Don't add user as a target of themselves
                        target_profiles_str += f'<target_profile name="{t_data.get("name")}">\nStance: {t_data.get("head_of_archive_stance")}\nHistory: {t_data.get("interaction_history")}\n</target_profile>\n'

                # 3. Agent 2: Synthesis with history, summary and persona
                bot_nickname = message.guild.me.display_name if message.guild else client.user.display_name
                final_response = await client.assistant.generate_refined_response(
                    query_text=query,
                    rag_response=str(rag_response),
                    history=previous_relevant,
                    summary=current_summary,
                    agent1_prompt=getattr(rag_response, 'agent1_prompt', ""),
                    bot_name=bot_nickname,
                    user_profile_str=user_profile_str,
                    target_profiles_str=target_profiles_str
                )
            except Exception as e:
                sys_logger.error(f"Error during RAG processing: {e}")
                # Use a thematic error message
                final_response = "⚠️ [System Error] Ошибка обработки запроса. Попробуйте позже."

        # 4. Send response
        sent_msg = await message.reply(final_response)

        # Add current transaction to internal history using server-specific nickname
        # (bot_nickname is already defined above in our shared scope)
        # Format user interaction
        user_interaction = f"{message.author.display_name}: {query}"
        # Format bot interaction
        bot_interaction = f"{bot_nickname}: {final_response}"
        
        chat_logger.info(f"User [{user_interaction}] -> Bot [{bot_interaction}]")
        
        # Async Task for Agent 3 (Social Chronicler)
        async def run_auditor():
            try:
                # Ensure we have rag_response if not error
                rag_str = str(rag_response) if 'rag_response' in locals() else "Сбои в матрице."
                op_data = await client.assistant.evaluate_interaction(
                    agent1_facts=rag_str,
                    agent2_response=final_response,
                    user_query=query,
                    user_name=message.author.display_name,
                    current_profile=user_profile_data
                )
                await client.opinions.update_user_opinion(
                    user_id=message.author.id,
                    name=message.author.display_name,
                    stance=op_data["stance"],
                    interaction=op_data["history"]
                )
            except Exception as e:
                sys_logger.error(f"Agent 3 task failed: {e}")
                
        asyncio.create_task(run_auditor())
        
        # Truncate strings before adding to history to prevent context bloat
        TRUNCATE_LIMIT = 700
        truncated_user = user_interaction if len(user_interaction) <= TRUNCATE_LIMIT else user_interaction[:TRUNCATE_LIMIT] + " [...]"
        truncated_bot = bot_interaction if len(bot_interaction) <= TRUNCATE_LIMIT else bot_interaction[:TRUNCATE_LIMIT] + " [...]"

        # Add current transaction to internal history
        client.history[channel_id] = previous_relevant + [truncated_user, truncated_bot]
        
        if len(client.history[channel_id]) >= 12:
            # We compress the 8 oldest messages and keep the 4 newest
            to_summarize = client.history[channel_id][:8]
            client.history[channel_id] = client.history[channel_id][8:]
            
            sys_logger.info(f"Post-response: Summarizing {len(to_summarize)} messages for channel {channel_id}...")
            try:
                new_summary = await client.assistant.generate_summary(current_summary, to_summarize)
                client.summaries[channel_id] = new_summary
                client._save_json(client.summaries_path, client.summaries)
                sys_logger.info(f"Summary updated for {channel_id}")
            except Exception as e:
                sys_logger.warning(f"Failed to update summary: {e}")
            
        client._save_json(client.history_path, client.history)
        return

    # if admin(Itadara) writes !export
    if message.content.startswith("!export") and str(message.author.id) == "470892009440149506": 
        sys_logger.info("Exporting chat...")
        await export_chat_to_json(channel, skip_id=message.id)
        return

client.run(os.getenv('TOKEN'))
