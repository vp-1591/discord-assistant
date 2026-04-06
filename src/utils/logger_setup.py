"""
Three-tiered logging configuration for the Discord assistant.

Log layers (per file):
  1. agent_traces.log       (INFO)   — Simple: ReAct steps, tool names, final answers
  2. agent_traces_detailed.log (DEBUG) — Detailed: prompts, headers, shadow thoughts, pre-reranked node text
  3. agent_traces_full.log  (TRACE=5) — Full: raw JSON I/O payloads from Ollama and vector/BM25 hits

All entries carry [TxID: XXXXXXXX] — a per-query Transaction ID set via
src/utils/context.py — so logs across all three files can be correlated
with a single grep.
"""
import logging
import os
import json
import queue
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from src.utils.context import transaction_id, span_id

# Custom numeric level below DEBUG(10) for raw JSON payloads
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class _TransactionFilter(logging.Filter):
    """Injects the current Transaction and Span ID into every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.tx_id = transaction_id.get()
        record.span_id = span_id.get()
        return True


class _ExactLevelFilter(logging.Filter):
    """Ensures a handler only processes logs of a specific level, ignoring higher ones."""
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


class _SimpleFormatter(logging.Formatter):
    """Compact, human-readable format for the INFO layer, with payload truncation support."""
    FMT = "%(asctime)s [TxID:%(tx_id)s][Span:%(span_id)s] %(levelname)s [%(name)s] %(message)s"

    def __init__(self):
        super().__init__(self.FMT, datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        original_msg = record.getMessage()
        limit = getattr(record, "truncate_simple", None)
        
        # Provide formatting without mutating the shared record dictionary
        msg_dict = record.__dict__.copy()
        if limit and len(original_msg) > limit:
            msg_dict['message'] = original_msg[:limit] + "\n...[TRUNCATED IN SIMPLE LOG]..."
        else:
            msg_dict['message'] = original_msg
            
        if self.usesTime():
            msg_dict['asctime'] = self.formatTime(record, self.datefmt)
            
        # Manually apply format to the safe dictionary
        return self._fmt % msg_dict


class _DetailedFormatter(logging.Formatter):
    """Extended format for DEBUG — preserves 'extra' payloads as indented blocks."""
    FMT = "%(asctime)s [TxID:%(tx_id)s][Span:%(span_id)s] %(levelname)s [%(name)s] %(message)s"
    _KNOWN_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def __init__(self):
        super().__init__(self.FMT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # Collect any extra keys the caller attached
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._KNOWN_ATTRS and k not in ("tx_id", "span_id", "message", "asctime")
        }
        if extras:
            try:
                extra_str = json.dumps(extras, ensure_ascii=False, indent=2, default=str)
            except Exception:
                extra_str = str(extras)
            return f"{base}\n{extra_str}"
        return base


class _FullFormatter(logging.Formatter):
    """Keeps full raw JSON payloads, appending the payload dict as a raw string."""
    FMT = "%(asctime)s [TxID:%(tx_id)s][Span:%(span_id)s] TRACE [%(name)s] %(message)s"
    _KNOWN_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def __init__(self):
        super().__init__(self.FMT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # Collect 'payload' or any other extras
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._KNOWN_ATTRS and k not in ("tx_id", "span_id", "message", "asctime")
        }
        if extras:
            # Multi-line indented JSON for better readability
            payload_str = json.dumps(extras, ensure_ascii=False, indent=4, default=str)
            return f"{base}\n{payload_str}"
        return base


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _make_handler(path: str, level: int, formatter: logging.Formatter,
                  max_bytes: int, backup_count: int, exact_level: bool = False) -> RotatingFileHandler:
    h = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count,
                            encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(formatter)
    if exact_level:
        h.addFilter(_ExactLevelFilter(level))
    return h


_listeners = []

def _setup_async_logger(name: str, level: int, handlers: list, filter_obj) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Apply filter synchronously before dispatching to background queue thread
    logger.addFilter(filter_obj)
    
    q = queue.Queue(-1)
    qh = QueueHandler(q)
    logger.addHandler(qh)
    
    listener = QueueListener(q, *handlers, respect_handler_level=True)
    listener.start()
    _listeners.append(listener)
    
    return logger


def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    _tx_filter = _TransactionFilter()

    # ------------------------------------------------------------------
    # 1. system.log — startup, ingestion, queue errors (INFO, 5 MB × 3)
    # ------------------------------------------------------------------
    sys_handler = _make_handler(
        os.path.join(log_dir, "system.log"),
        logging.INFO, _SimpleFormatter(),
        max_bytes=5 * 1024 * 1024, backup_count=3
    )
    sys_logger = _setup_async_logger("system", logging.INFO, [sys_handler], _tx_filter)

    # ------------------------------------------------------------------
    # 2. chat_history.log — discord user↔bot messages (INFO, 2 MB × 3)
    # ------------------------------------------------------------------
    chat_handler = _make_handler(
        os.path.join(log_dir, "chat_history.log"),
        logging.INFO, _SimpleFormatter(),
        max_bytes=2 * 1024 * 1024, backup_count=3
    )
    chat_logger = _setup_async_logger("chat_history", logging.INFO, [chat_handler], _tx_filter)

    # ------------------------------------------------------------------
    # 3. indexing_summarization.log — ingest prompts/responses (INFO, 10 MB × 2)
    # ------------------------------------------------------------------
    indexing_handler = _make_handler(
        os.path.join(log_dir, "indexing_summarization.log"),
        logging.INFO, _SimpleFormatter(),
        max_bytes=10 * 1024 * 1024, backup_count=2
    )
    indexing_logger = _setup_async_logger("indexing", logging.INFO, [indexing_handler], _tx_filter)

    # ------------------------------------------------------------------
    # 4a. agent_traces.log — Layer 1: simple ReAct timeline (INFO, 5 MB × 3)
    # ------------------------------------------------------------------
    trace_simple_h = _make_handler(
        os.path.join(log_dir, "agent_traces.log"),
        logging.INFO, _SimpleFormatter(),
        max_bytes=5 * 1024 * 1024, backup_count=3
    )

    # ------------------------------------------------------------------
    # 4b. agent_traces_detailed.log — Layer 2: prompts, thoughts, pre-rerank text (DEBUG, 20 MB × 2)
    # ------------------------------------------------------------------
    trace_detailed_h = _make_handler(
        os.path.join(log_dir, "agent_traces_detailed.log"),
        logging.DEBUG, _DetailedFormatter(),
        max_bytes=20 * 1024 * 1024, backup_count=2
    )

    # ------------------------------------------------------------------
    # 4c. agent_traces_full.log — Layer 3: raw JSON I/O (TRACE=5, 10 MB × 1)
    # ------------------------------------------------------------------
    trace_full_h = _make_handler(
        os.path.join(log_dir, "agent_traces_full.log"),
        TRACE, _FullFormatter(),
        max_bytes=10 * 1024 * 1024, backup_count=1,
        exact_level=True
    )

    trace_logger = _setup_async_logger("agent_traces", TRACE, 
                                       [trace_simple_h, trace_detailed_h, trace_full_h], 
                                       _tx_filter)

    return sys_logger, trace_logger, chat_logger, indexing_logger


# Suppress noisy third-party loggers
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("discord").setLevel(logging.INFO)

# Initialise once on import and export
sys_logger, trace_logger, chat_logger, indexing_logger = setup_logger()

import atexit
atexit.register(lambda: [listener.stop() for listener in _listeners])


# ---------------------------------------------------------------------------
# Convenience: emit a TRACE-level record
# ---------------------------------------------------------------------------

def log_trace(logger: logging.Logger, msg: str, **extra):
    """Log raw JSON payload at TRACE level (below DEBUG)."""
    if logger.isEnabledFor(TRACE):
        logger.log(TRACE, msg, extra=extra)
