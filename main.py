import discord
import os
from dotenv import load_dotenv
from src.data.export_chat import export_chat_to_json, resolve_mentions
import asyncio

import sys
from src.core.run_llama_index import RAGAssistant
from src.utils.logger_setup import sys_logger, chat_logger
from src.data.opinion_manager import OpinionManager
from src.data.history_manager import HistoryManager
from src.config.config import SUMMARIES_PATH, HISTORY_PATH, PERSIST_DIR, ADMIN_IDS

load_dotenv()

# Global exception handler to ensure all errors go to system.log
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    sys_logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents = discord.Intents.all())
        self.assistant = None
        self._assistant_loading = False
        self.history_manager = HistoryManager(SUMMARIES_PATH, HISTORY_PATH)
        self.opinions = OpinionManager()
        self.message_queue = asyncio.Queue()
        self.processing_task = None

    async def setup_assistant(self):
        if self.assistant is None and not self._assistant_loading:
            self._assistant_loading = True
            try:
                id_map = {}
                if not os.path.exists(PERSIST_DIR):
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
                
                if self.processing_task is None:
                    self.processing_task = asyncio.create_task(self.process_message_queue())# Message queue processor
            finally:
                self._assistant_loading = False

    async def process_message_queue(self):
        """Continuously pulls messages from the queue and processes them sequentially."""
        while True:
            message = await self.message_queue.get()
            try:
                await self._handle_message_internal(message)
            except Exception as e:
                sys_logger.error(f"Error processing message from queue: {e}")
            finally:
                self.message_queue.task_done()

    async def _handle_message_internal(self, message):
        query = message.content.replace(f'<@{self_id}>', '').replace(f'<@!{self_id}>', '').strip()
        query = resolve_mentions(message, text=query)
        if not query: query = "Привет!"

        sys_logger.info(f"Mentions detected. Query: {query}")
        
        channel = message.channel
        channel_id = str(channel.id)

        async with channel.typing():
            history = self.history_manager.get_history(channel_id)
            summary = self.history_manager.get_summary(channel_id)
            
            replied_to_msg = None
            if message.reference and message.reference.message_id:
                try:
                    ref_msg = await channel.fetch_message(message.reference.message_id)
                    replied_to_msg = f"{ref_msg.author.display_name}: {ref_msg.content}"
                    sys_logger.info(f"Replied-to message found: {replied_to_msg}")
                except Exception as e:
                    sys_logger.warning(f"Could not fetch referenced message: {e}")

            try:
                bot_nickname = message.guild.me.display_name if message.guild else self.user.display_name
                final_response = await self.assistant.generate_refined_response(
                    query_text=query,
                    history=history,
                    summary=summary,
                    bot_name=bot_nickname,
                    author_id=str(message.author.id),
                    author_name=message.author.display_name,
                    replied_to_msg=replied_to_msg
                )
            except Exception as e:
                sys_logger.error(f"Error during RAG processing: {e}")
                final_response = "⚠️ [System Error] Ошибка обработки запроса. Попробуйте позже."

        await message.reply(final_response)

        # Update history
        user_interaction = f"{message.author.display_name}: {query}"
        bot_interaction = f"{bot_nickname}: {final_response}"
        chat_logger.info(f"User [{user_interaction}] -> Bot [{bot_interaction}]")
        
        self.history_manager.add_to_history(channel_id, [user_interaction, bot_interaction])
        
        # Summarization logic
        to_summarize = self.history_manager.truncate_history(channel_id)
        if to_summarize:
            sys_logger.info(f"Post-response: Summarizing for channel {channel_id}...")
            try:
                new_summary = await self.assistant.generate_summary(summary, to_summarize)
                self.history_manager.update_summary(channel_id, new_summary)
                sys_logger.info(f"Summary updated for {channel_id}")
            except Exception as e:
                sys_logger.warning(f"Failed to update summary: {e}")

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

    # Export command
    if message.content.startswith("!export") and str(message.author.id) in ADMIN_IDS: 
        sys_logger.info("Exporting chat...")
        new_msgs_count = await export_chat_to_json(message.channel, skip_id=message.id)
        
        if new_msgs_count > 0 and client.assistant is not None:
             sys_logger.info(f"Triggering Vector DB update for {new_msgs_count} new messages...")
             nodes_added = await asyncio.to_thread(client.assistant.update_index)
             sys_logger.info(f"Vector DB Update Complete. Inserted {nodes_added} new chunked nodes.")
        else:
             sys_logger.info("Export finished. No new messages found for indexing.")
             
        return
    if client.user.mentioned_in(message):
        await client.message_queue.put(message)
        sys_logger.info(f"Message added to queue. Queue size: {client.message_queue.qsize()}")

client.run(os.getenv('TOKEN'))
