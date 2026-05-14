import asyncio
import os
from typing import Any

import chromadb

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ChromaDB client cache - one client per path
_chroma_clients: dict[str, Any] = {}


def get_chroma_client(db_path: str) -> Any:
    """Get or create a ChromaDB client for the specified path."""
    path = db_path
    if path not in _chroma_clients:
        _chroma_clients[path] = chromadb.PersistentClient(path=path)
    return _chroma_clients[path]


def get_or_create_collection(name: str, db_path: str):
    """Get or create a ChromaDB collection by name."""
    client = get_chroma_client(db_path)
    return client.get_or_create_collection(name=name)


async def _delete_entries_by_filepath(filepath: str, coll) -> int:
    """Delete all indexed entries for a given filepath from a collection."""
    abs_path = os.path.abspath(filepath)

    def _delete():
        result = coll.delete(where={"filepath": {"$eq": abs_path}})
        return result.get("deleted", 0) if isinstance(result, dict) else 0

    return await asyncio.to_thread(_delete)
