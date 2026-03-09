import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Standard format: [Timestamp] [Level] [LoggerName] message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

    # 1. System Logger (General flow, errors, locks)
    system_handler = RotatingFileHandler(
        os.path.join(log_dir, "system.log"), 
        maxBytes=5*1024*1024, # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    system_handler.setFormatter(formatter)
    
    sys_logger = logging.getLogger("system")
    sys_logger.setLevel(logging.INFO)
    sys_logger.addHandler(system_handler)

    # 2. Agent Traces (Deep dive into Prompt/Response)
    trace_handler = RotatingFileHandler(
        os.path.join(log_dir, "agent_traces.log"),
        maxBytes=10*1024*1024, # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    trace_handler.setFormatter(formatter)
    
    trace_logger = logging.getLogger("agent_traces")
    trace_logger.setLevel(logging.INFO)
    trace_logger.addHandler(trace_handler)

    # 3. Chat History (Clean human-readable conversation log)
    chat_handler = RotatingFileHandler(
        os.path.join(log_dir, "chat_history.log"),
        maxBytes=2*1024*1024, # 2MB
        backupCount=3,
        encoding="utf-8"
    )
    chat_handler.setFormatter(formatter)
    
    chat_logger = logging.getLogger("chat_history")
    chat_logger.setLevel(logging.INFO)
    chat_logger.addHandler(chat_handler)

    return sys_logger, trace_logger, chat_logger

# Optional: suppress some verbose third-party logging if needed
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("discord").setLevel(logging.INFO)

# Run once at import and export loggers
sys_logger, trace_logger, chat_logger = setup_logger()
