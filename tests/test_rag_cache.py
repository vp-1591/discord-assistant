import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.rag_cache import RAGCache

async def test_cache():
    print("Starting RAG Cache Tests...")
    cache = RAGCache(capacity=3) # Use smaller capacity for testing
    
    # 1. Test empty cache
    print("Test 1: Empty cache")
    assert cache.get_recent_queries() == []
    print("Pass")
    
    # 2. Test storing and retrieving
    print("Test 2: Store and retrieve")
    cache.store("query 1", "response 1")
    queries = cache.get_recent_queries()
    assert len(queries) == 1
    assert queries[0]["query"] == "query 1"
    assert cache.get_result_by_id(1) == "response 1"
    print("Pass")
    
    # 3. Test LRU Eviction
    print("Test 3: LRU Eviction")
    cache.store("query 2", "response 2")
    cache.store("query 3", "response 3")
    cache.store("query 4", "response 4") # Should evict "query 1"
    
    queries = cache.get_recent_queries()
    print(f"Queries after 4th insert: {queries}")
    assert len(queries) == 3
    # Our queries return in order of insertion. index 0 is first inserted.
    # Actually my implementation returns [{"id": 1, "query": q1}, {"id": 2, "query": q2}] where index relates to OrderDict keys.
    # OrderedDict popitem(last=False) pops the FIRST one.
    # So if we have [q1, q2, q3] and add q4, we get [q2, q3, q4].
    assert queries[0]["query"] == "query 2"
    assert queries[2]["query"] == "query 4"
    print("Pass")
    
    # 4. Test re-access moving to end
    print("Test 4: Access moves to end")
    # Current: [q2, q3, q4]
    # Accessing q2 should move it to end?
    # In my get_result_by_id: self.cache.move_to_end(query)
    # So [q3, q4, q2]
    res = cache.get_result_by_id(1) 
    assert res == "response 2"
    queries = cache.get_recent_queries()
    # Now query 2 should be at index 2 (last)
    assert queries[2]["query"] == "query 2"
    assert queries[0]["query"] == "query 3"
    print("Pass")
    
    print("All Cache Logic Tests Passed!")

if __name__ == "__main__":
    asyncio.run(test_cache())
