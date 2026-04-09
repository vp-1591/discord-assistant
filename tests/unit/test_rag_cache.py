import pytest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.rag_cache import RAGCache

@pytest.fixture
def cache(tmp_path):
    persist_file = tmp_path / "test_cache.json"
    return RAGCache(persist_path=str(persist_file), capacity=3)

@pytest.mark.asyncio
async def test_empty_cache(cache):
    assert cache.get_recent_queries() == []

@pytest.mark.asyncio
async def test_store_and_retrieve(cache):
    cache.store("query 1", "response 1")
    queries = cache.get_recent_queries()
    assert len(queries) == 1
    assert queries[0]["query"] == "query 1"
    
    stored_id = queries[0]["id"]
    assert cache.get_result_by_id(stored_id) == "response 1"

@pytest.mark.asyncio
async def test_lru_eviction(cache):
    cache.store("query 1", "response 1")
    cache.store("query 2", "response 2")
    cache.store("query 3", "response 3")
    cache.store("query 4", "response 4") # Evicts query 1

    queries = cache.get_recent_queries()
    assert len(queries) == 3
    assert queries[0]["query"] == "query 2"
    assert queries[2]["query"] == "query 4"

@pytest.mark.asyncio
async def test_access_moves_to_end(cache):
    cache.store("query 1", "response 1")
    cache.store("query 2", "response 2")
    cache.store("query 3", "response 3")
    
    queries_before = cache.get_recent_queries()
    query_1_id = next(q["id"] for q in queries_before if q["query"] == "query 1")
    
    res = cache.get_result_by_id(query_1_id)
    assert res == "response 1"
    
    queries = cache.get_recent_queries()
    assert queries[2]["query"] == "query 1"  # Moved to end
    assert queries[0]["query"] == "query 2"
