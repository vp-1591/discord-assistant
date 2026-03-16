import discord
import os
from dotenv import load_dotenv
from src.data.export_chat import export_chat_to_json, resolve_mentions
import asyncio

from src.core.run_llama_index import RAGAssistant
from src.utils.logger_setup import sys_logger, chat_logger
from src.data.opinion_manager import OpinionManager
from src.data.history_manager import HistoryManager
from src.config.config import SUMMARIES_PATH, HISTORY_PATH, FORCE_REBUILD, PERSIST_DIR

load_dotenv()

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents = discord.Intents.all())
        self.assistant = None
        self._assistant_loading = False
        self.history_manager = HistoryManager(SUMMARIES_PATH, HISTORY_PATH)
        self.opinions = OpinionManager()

    async def setup_assistant(self):
        if self.assistant is None and not self._assistant_loading:
            self._assistant_loading = True
            try:
                id_map = {}
                if FORCE_REBUILD or not os.path.exists(PERSIST_DIR):
                    print("🤖 Rebuild required. Collecting latest names and roles...")
                    for user in self.users: id_map[str(user.id)] = user.display_name
                    for guild in self.guilds:
                        for role in guild.roles: id_map[str(role.id)] = role.name
                        for member in guild.members: id_map[str(member.id)] = member.display_name
                    print(f"✅ Map built with {len(id_map)} identities.")
                else:
                    print("📦 Loading existing index (skipping identity scan).")

                bot_name = self.guilds[0].me.display_name if self.guilds else self.user.display_name
                print(f"🤖 Initializing RAG Assistant as '{bot_name}'...")

                self.assistant = await asyncio.to_thread(
                    RAGAssistant, id_map=id_map, name=bot_name, opinion_manager=self.opinions
                )
                print(f"✅ RAG Assistant ready.")
            finally:
                self._assistant_loading = False

self_id = 1208704829665447947
client = aclient()

@client.event
async def on_ready():
    sys_logger.info("Connecting...")
    await client.setup_assistant()
    sys_logger.info(f"Connected as {client.user}")

@client.event
async def on_message(message):
    if message.author.id == self_id or message.author.bot: return 
    
    # Check if bot is mentioned
    if client.user.mentioned_in(message):
        query = message.content.replace(f'<@{self_id}>', '').replace(f'<@!{self_id}>', '').strip()
        query = resolve_mentions(message, text=query)
        if not query: query = "Привет!"

        sys_logger.info(f"Mentions detected. Query: {query}")
        
        channel = message.channel
        channel_id = str(channel.id)

        async with channel.typing():
            history = client.history_manager.get_history(channel_id)
            summary = client.history_manager.get_summary(channel_id)
            
            try:
                bot_nickname = message.guild.me.display_name if message.guild else client.user.display_name
                final_response = await client.assistant.generate_refined_response(
                    query_text=query,
                    history=history,
                    summary=summary,
                    bot_name=bot_nickname,
                    author_id=str(message.author.id),
                    author_name=message.author.display_name,
                )
            except Exception as e:
                sys_logger.error(f"Error during RAG processing: {e}")
                final_response = "⚠️ [System Error] Ошибка обработки запроса. Попробуйте позже."

        await message.reply(final_response)

        # Update history
        user_interaction = f"{message.author.display_name}: {query}"
        bot_interaction = f"{bot_nickname}: {final_response}"
        chat_logger.info(f"User [{user_interaction}] -> Bot [{bot_interaction}]")
        
        truncated_user = HistoryManager.truncate_string(user_interaction)
        truncated_bot = HistoryManager.truncate_string(bot_interaction)
        client.history_manager.add_to_history(channel_id, [truncated_user, truncated_bot])
        
        # Summarization logic
        to_summarize = client.history_manager.truncate_history(channel_id)
        if to_summarize:
            sys_logger.info(f"Post-response: Summarizing for channel {channel_id}...")
            try:
                new_summary = await client.assistant.generate_summary(summary, to_summarize)
                client.history_manager.update_summary(channel_id, new_summary)
                sys_logger.info(f"Summary updated for {channel_id}")
            except Exception as e:
                sys_logger.warning(f"Failed to update summary: {e}")
        return

    # Export command
    if message.content.startswith("!export") and str(message.author.id) == "470892009440149506": 
        sys_logger.info("Exporting chat...")
        await export_chat_to_json(message.channel, skip_id=message.id)
        return

client.run(os.getenv('TOKEN'))
