import asyncio
import os
from typing import Any

import httpx

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_EMBED_MODEL = "nvidia/nv-embedcode-7b-v1"
DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"

# Rate limiting semaphore for concurrent API calls
api_semaphore = asyncio.Semaphore(5)

# Global HTTP client with connection pooling for improved async performance
_http_client: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=20),
            timeout=30.0,
        )
    return _http_client


def get_rerank_url(model: str) -> str:
    """Get the API URL for a given rerank model."""
    if model == "nvidia/rerank-qa-mistral-4b":
        return "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    # Strip vendor prefix for URL path
    model_slug = (
        model.split("/")[-1].replace(".", "_")
        if "/" in model
        else model.replace(".", "_")
    )
    return f"https://ai.api.nvidia.com/v1/retrieval/nvidia/{model_slug}/reranking"


async def get_embeddings_batch(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    input_type: str = "passage",
    truncate: str = "NONE",
) -> list[list[float]]:
    """Get embedding vectors from NVIDIA NIM API for a batch of texts."""
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is required")

    if not texts:
        return []

    payload = {
        "input": texts,
        "model": model,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": truncate,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    client = await _get_http_client()
    async with api_semaphore:
        response = await client.post(NIM_EMBED_URL, json=payload, headers=headers)

    if response.status_code != 200:
        error_msg = (
            f"API request failed with status {response.status_code}: {response.text}"
        )
        raise ValueError(error_msg)

    result = response.json()

    if "data" not in result or not result["data"]:
        raise ValueError(f"Empty response from NVIDIA NIM API: {result}")

    # Sort by index to ensure order matches input
    sorted_data = sorted(result["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


async def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    input_type: str = "passage",
    truncate: str = "NONE",
) -> list[float]:
    """Get embedding vector from NVIDIA NIM API for a single text."""
    embeddings = await get_embeddings_batch(
        [text], model=model, input_type=input_type, truncate=truncate
    )
    return embeddings[0]


async def rerank(
    query: str,
    documents: list[str],
    model: str = DEFAULT_RERANK_MODEL,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Rerank documents by relevance to query."""
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is required")

    if not documents:
        return []

    payload = {
        "model": model,
        "query": {"text": query},
        "passages": [{"text": doc} for doc in documents],
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    rerank_url = get_rerank_url(model)
    client = await _get_http_client()
    async with api_semaphore:
        response = await client.post(rerank_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("rankings", [])
