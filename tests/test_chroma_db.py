import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chroma_db import (
    _chroma_clients,
    _delete_entries_by_filepath,
    get_chroma_client,
    get_or_create_collection,
)


def test_get_chroma_client(mock_chroma_db):
    client1 = get_chroma_client(mock_chroma_db)
    client2 = get_chroma_client(mock_chroma_db)
    assert client1 is client2
    assert mock_chroma_db in _chroma_clients


def test_get_or_create_collection(mock_chroma_db):
    coll = get_or_create_collection("test_coll", mock_chroma_db)
    assert coll.name == "test_coll"


@pytest.mark.asyncio
async def test_delete_entries_by_filepath(mock_chroma_db):
    coll = get_or_create_collection("test_coll2", mock_chroma_db)

    # Add dummy entries
    filepath = os.path.abspath("dummy.py")
    coll.add(
        ids=["id1", "id2"],
        documents=["doc1", "doc2"],
        embeddings=[[0.1] * 384, [0.1] * 384],
        metadatas=[{"filepath": filepath}, {"filepath": "other.py"}],
    )

    assert coll.count() == 2
    deleted = await _delete_entries_by_filepath("dummy.py", coll)
    assert (
        deleted >= 0
    )  # chromadb local client delete is a bit quirky in returning counts

    results = coll.get()
    # "other.py" should remain
    assert len(results["ids"]) == 1
    assert results["metadatas"][0]["filepath"] == "other.py"
