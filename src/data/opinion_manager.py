import json
import os
import asyncio
import logging
from difflib import get_close_matches

logger = logging.getLogger("system")

class OpinionManager:
    def __init__(self, file_path="cache/opinions.json"):
        self.file_path = file_path
        self.lock = asyncio.Lock()
        self.opinions = self._load_opinions()
        
    def _load_opinions(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load opinions.json: {e}")
                return {}
        return {}

    async def save_opinions(self):
        async with self.lock:
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                # Read latest from disk then update to avoid losing data from parallel edits if any exist outside this instance
                # Though within this process, self.opinions stays updated.
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(self.opinions, f, ensure_ascii=False, indent=4)
                logger.info("Saved opinions.json successfully.")
            except Exception as e:
                logger.error(f"Failed to save opinions.json: {e}")

    def get_known_names(self):
        """Returns a list of display names we have opinions for."""
        return [data.get("name") for data in self.opinions.values() if data.get("name")]

    def find_targets(self, text, threshold=0.6):
        """
        Uses fuzzy matching to find known users mentioned in the text.
        Returns a list of (discord_id, data) for matched users.
        """
        known_names = self.get_known_names()
        if not known_names:
            return []
            
        words = text.split()
        matches = []
        seen_ids = set()

        for word in words:
            # Clean punctuation
            clean_word = "".join(filter(str.isalnum, word))
            if not clean_word: continue
            
            close_matches = get_close_matches(clean_word, known_names, n=1, cutoff=threshold)
            if close_matches:
                matched_name = close_matches[0]
                # Find the ID for this name
                for uid, data in self.opinions.items():
                    if data.get("name") == matched_name and uid not in seen_ids:
                        matches.append((uid, data))
                        seen_ids.add(uid)
        return matches

    def get_user_profile(self, user_id, default_name="Unknown"):
        profile = self.opinions.get(str(user_id))
        if not profile:
            return None
        return profile

    async def update_user_opinion(self, user_id, name, stance, interaction):
        """
        Updates or creates a user's opinion entry. 
        Note: We reload the data inside the lock to ensure we don't overwrite 
        intervening updates from other background tasks.
        """
        async with self.lock:
            # Refresh from disk to merge possible changes from other concurrent tasks
            self.opinions = self._load_opinions()
            
            user_id_str = str(user_id)
            self.opinions[user_id_str] = {
                "name": name,
                "head_of_archive_stance": stance,
                "interaction_history": interaction
            }
            
            # Write back
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(self.opinions, f, ensure_ascii=False, indent=4)
                logger.info(f"Updated opinion for user {name} ({user_id_str})")
            except Exception as e:
                logger.error(f"Failed to update opinion for {name}: {e}")
