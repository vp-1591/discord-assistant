"""
Transaction context for cross-layer log correlation.

Usage:
  - Set at the start of each user query: `transaction_id.set(new_tx_id())`
  - Use `run_with_context(sync_fn, *args)` instead of bare calls inside
    `asyncio.to_thread` to guarantee the Transaction_ID propagates into
    background OS threads (LlamaIndex DB tools, Reranker, etc.)
"""
import uuid
import contextvars

# Holds the current transaction ID. Default 'SYS' means the log came from
# outside a named user query (startup, ingestion pipeline, etc.)
transaction_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "transaction_id", default="SYS"
)

span_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "span_id", default="----"
)


def new_tx_id() -> str:
    """Generate a short 8-character unique transaction ID."""
    return uuid.uuid4().hex[:8].upper()

def new_span_id() -> str:
    """Generate a 4-character unique span ID for sub-operations."""
    return uuid.uuid4().hex[:4].upper()

def run_with_context(func, *args, **kwargs):
    """
    Thread-safe context propagation wrapper.

    asyncio.to_thread on Python 3.9+ *does* copy the current Context
    automatically. This wrapper uses copy_context().run() as an explicit
    guarantee for older thread pool internals inside LlamaIndex (e.g.
    BM25Retriever, SQLite tools) that may spawn threads through other means.

    Usage:
        await asyncio.to_thread(run_with_context, my_sync_fn, arg1, arg2)
    """
    ctx = contextvars.copy_context()
    return ctx.run(func, *args, **kwargs)
