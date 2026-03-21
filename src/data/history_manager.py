import os
import json
from typing import Dict, List
from src.utils.logger_setup import sys_logger

class HistoryManager:
    def __init__(self, summaries_path: str, history_path: str):
        self.summaries_path = summaries_path
        self.history_path = history_path
        self.summaries = self._load_json(self.summaries_path)
        self.history = self._load_json(self.history_path)

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                sys_logger.error(f"Failed to load {path}: {e}")
                return {}
        return {}

    def _save_json(self, path: str, data: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            sys_logger.error(f"Failed to save {path}: {e}")

    def get_history(self, channel_id: str) -> List[str]:
        if channel_id not in self.history:
            sys_logger.info(f"Initialized empty local history for channel {channel_id}.")
            self.history[channel_id] = []
            self.save_history()
        return self.history.get(channel_id, [])

    def add_to_history(self, channel_id: str, messages: List[str]):
        current = self.get_history(channel_id)
        self.history[channel_id] = current + messages
        self.save_history()

    def truncate_history(self, channel_id: str, limit: int = 12, keep: int = 4) -> List[str]:
        history = self.get_history(channel_id)
        if len(history) >= limit:
            to_summarize = history[:limit - keep]
            self.history[channel_id] = history[limit - keep:]
            self.save_history()
            return to_summarize
        return []

    def get_summary(self, channel_id: str) -> str:
        return self.summaries.get(channel_id)

    def update_summary(self, channel_id: str, new_summary: str):
        self.summaries[channel_id] = new_summary
        self.save_summaries()

    def save_history(self):
        self._save_json(self.history_path, self.history)

    def save_summaries(self):
        self._save_json(self.summaries_path, self.summaries)
