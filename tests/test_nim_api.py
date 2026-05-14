import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import nim_api
from nim_api import (
    _get_http_client,
    get_embedding,
    get_embeddings_batch,
    get_rerank_url,
    rerank,
)


def test_get_rerank_url():
    url = get_rerank_url("nvidia/rerank-qa-mistral-4b")
    assert url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    url2 = get_rerank_url("nvidia/llama-nemotron-rerank-1b-v2")
    assert (
        url2
        == "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
    )


@pytest.mark.asyncio
async def test_get_http_client():
    client = await _get_http_client()
    assert isinstance(client, httpx.AsyncClient)
    assert client.is_closed is False
    client2 = await _get_http_client()
    assert client is client2


@pytest.mark.asyncio
async def test_get_embeddings_batch(mocker):
    mock_post = mocker.AsyncMock()
    mock_post.return_value.status_code = 200
    mock_json = mocker.MagicMock(
        return_value={
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
    )
    mock_post.return_value.json = mock_json

    mock_client = mocker.MagicMock()
    mock_client.post = mock_post
    mocker.patch("nim_api._get_http_client", return_value=mock_client)

    embeddings = await get_embeddings_batch(["test1", "test2"])
    assert len(embeddings) == 2
    # Check that sorting by index happened
    assert embeddings[0] == [0.1, 0.2]
    assert embeddings[1] == [0.3, 0.4]


@pytest.mark.asyncio
async def test_get_embeddings_batch_empty():
    embeddings = await get_embeddings_batch([])
    assert embeddings == []


@pytest.mark.asyncio
async def test_get_embeddings_batch_no_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    nim_api.NVIDIA_API_KEY = ""
    with pytest.raises(
        ValueError, match="NVIDIA_API_KEY environment variable is required"
    ):
        await get_embeddings_batch(["test"])
    # Reset for other tests
    nim_api.NVIDIA_API_KEY = "test_api_key"


@pytest.mark.asyncio
async def test_get_embeddings_batch_error(mocker):
    mock_post = mocker.AsyncMock()
    mock_post.return_value.status_code = 400
    mock_post.return_value.text = "Bad Request"

    mock_client = mocker.MagicMock()
    mock_client.post = mock_post
    mocker.patch("nim_api._get_http_client", return_value=mock_client)

    with pytest.raises(ValueError, match="API request failed with status 400"):
        await get_embeddings_batch(["test"])


@pytest.mark.asyncio
async def test_get_embeddings_batch_empty_response(mocker):
    mock_post = mocker.AsyncMock()
    mock_post.return_value.status_code = 200
    mock_json = mocker.MagicMock(return_value={})
    mock_post.return_value.json = mock_json

    mock_client = mocker.MagicMock()
    mock_client.post = mock_post
    mocker.patch("nim_api._get_http_client", return_value=mock_client)

    with pytest.raises(ValueError, match="Empty response"):
        await get_embeddings_batch(["test"])


@pytest.mark.asyncio
async def test_get_embedding(mocker):
    mocker.patch("nim_api.get_embeddings_batch", return_value=[[0.5, 0.6]])
    emb = await get_embedding("test")
    assert emb == [0.5, 0.6]


@pytest.mark.asyncio
async def test_rerank(mocker):
    mock_post = mocker.AsyncMock()
    mock_post.return_value.raise_for_status = mocker.MagicMock()
    mock_json = mocker.MagicMock(
        return_value={"rankings": [{"index": 0, "logit": 5.0}]}
    )
    mock_post.return_value.json = mock_json

    mock_client = mocker.MagicMock()
    mock_client.post = mock_post
    mocker.patch("nim_api._get_http_client", return_value=mock_client)

    results = await rerank("query", ["doc1"])
    assert len(results) == 1
    assert results[0]["logit"] == 5.0


@pytest.mark.asyncio
async def test_rerank_empty():
    results = await rerank("query", [])
    assert results == []


@pytest.mark.asyncio
async def test_rerank_no_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    nim_api.NVIDIA_API_KEY = ""
    with pytest.raises(
        ValueError, match="NVIDIA_API_KEY environment variable is required"
    ):
        await rerank("query", ["doc1"])
    nim_api.NVIDIA_API_KEY = "test_api_key"
