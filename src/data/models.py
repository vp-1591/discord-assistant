"""
SQLAlchemy models for the local Discord data store.

The `Message` table is intentionally denormalized: each row corresponds to one
raw Discord message.  Multiple rows share the same `chunk_id`, which ties them
to the LlamaIndex TextNode whose summary was built from that group of messages.
Use `chunk_id` to join back from a retrieved summary node to the original text.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # The real Discord snowflake ID for this message.
    message_id = Column(String(32), nullable=False, unique=True)

    # Links this row to the LlamaIndex TextNode that summarised its chunk.
    # Many rows share the same chunk_id (one chunk = many messages).
    chunk_id = Column(String(128), nullable=True, index=True)

    author = Column(String(256), nullable=True)

    # Full raw message text.
    content = Column(Text, nullable=True)

    # ISO-8601 string: "2024-03-26T12:00:00"  — compatible with SQLite date funcs.
    timestamp = Column(String(32), nullable=True)

    channel = Column(String(128), nullable=True)


# Engine targeting discord_data.db in the project root (resolved at runtime).
engine = create_engine("sqlite:///discord_data.db", echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
