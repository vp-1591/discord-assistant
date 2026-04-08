import collections
import json
import os
from typing import Optional

class RAGCache:
    """
    A simple LRU cache for storing RAG search results.
    Maintains a maximum of 5 most recent search results and persists to disk.
    """
    def __init__(self, persist_path: str, capacity: int = 5):
        self.capacity = capacity
        self.persist_path = persist_path
        self.cache = self._load_cache()

    def _load_cache(self) -> collections.OrderedDict:
        """Loads the cache from disk if it exists."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert list of pairs back to OrderedDict
                    return collections.OrderedDict(data)
            except Exception as e:
                print(f"Error loading RAG cache: {e}")
        return collections.OrderedDict()

    def _save_cache(self):
        """Saves the cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                # Store as list of pairs to preserve order in JSON
                json.dump(list(self.cache.items()), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving RAG cache: {e}")

    def store(self, query: str, response: str):
        """Stores a query and its response. If query exists, it moves it to the end."""
        if query in self.cache:
            self.cache.move_to_end(query)
        self.cache[query] = response
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        self._save_cache()

    def get_recent_queries(self) -> list:
        """Returns a list of recent queries with their IDs (1-based index)."""
        queries = list(self.cache.keys())
        return [{"id": i + 1, "query": q} for i, q in enumerate(queries)]

    def get(self, query: str) -> Optional[str]:
        """Retrieves a result by exact query string. Returns None on miss."""
        if query in self.cache:
            self.cache.move_to_end(query)
            self._save_cache()
            return self.cache[query]
        return None

    def get_result_by_id(self, result_id: int) -> str:
        """Retrieves a specific result by its ID."""
        queries = list(self.cache.keys())
        if 1 <= result_id <= len(queries):
            result = self.get(queries[result_id - 1])
            return result if result is not None else f"Error: No cached result found for ID {result_id}."
        return f"Error: No cached result found for ID {result_id}."

    def clear(self):
        """Clears the cache and deletes the file."""
        self.cache.clear()
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)
