#!/usr/bin/env python3
"""
NVIDIA NIM MCP Server - Code Embedding and Reranking Server

This MCP server provides code embedding and reranking capabilities using NVIDIA NIM API.
It uses ChromaDB as the vector database for storing and retrieving embeddings.
"""

# Ensure local imports work when running from any directory
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Standard library imports
import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
import chromadb
import httpx

# Global HTTP client with connection pooling for improved async performance
_http_client: Optional[httpx.AsyncClient] = None


async def _get_http_client() -> httpx.AsyncClient:
    """Get or create the global HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=20),
            timeout=30.0,
        )
    return _http_client


# MCP imports
from mcp.server.fastmcp import FastMCP

# Local imports - AST chunker for semantic code splitting
from chunker import (
    chunk_code_by_ast,
    format_for_nvidia_nim,
    get_supported_extensions,
    is_extension_supported,
)

# ============================================================================
# Configuration
# ============================================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Rate limiting semaphore for concurrent API calls
api_semaphore = asyncio.Semaphore(5)


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


# Default models
DEFAULT_EMBED_MODEL = "nvidia/nv-embedcode-7b-v1"
DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"

# ChromaDB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# File indexing settings
MAX_FILE_SIZE = 1024 * 1024  # 1MB


# AST Chunking settings
ENABLE_AST_CHUNKING = True  # Enable AST-based semantic chunking
MAX_CHUNK_SIZE = 4096  # Maximum chunk size in characters before splitting

# Supported languages and file extensions
SUPPORTED_EXTENSIONS = {
    "python": [".py", ".pyw", ".pyi"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"],
    "c": [".c", ".h"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "scala": [".scala"],
    "shell": [".sh", ".bash", ".zsh"],
    "sql": [".sql"],
    "html": [".html", ".htm"],
    "css": [".css"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "markdown": [".md", ".markdown"],
    "lua": [".lua"],
    "r": [".r"],
    "matlab": [".m"],
    "perl": [".pl"],
    "haskell": [".hs"],
    "elixir": [".ex", ".exs"],
}

# Reverse mapping for language detection
EXTENSION_TO_LANGUAGE = {}
for language, extensions in SUPPORTED_EXTENSIONS.items():
    for ext in extensions:
        EXTENSION_TO_LANGUAGE[ext] = language

# Skip directories and files
SKIP_DIRS = {
    "node_modules",
    "vendor",
    ".git",
    ".github",
    ".vscode",
    ".vs",
    "bin",
    "obj",
    "out",
    ".next",
    ".cache",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}

# Initialize MCP server
mcp = FastMCP("nim-code-embed-rerank")

# ChromaDB client cache - one client per path
# Type annotation uses ClientAPI from chromadb
_chroma_clients: Dict[str, Any] = {}


def get_chroma_client(db_path: str) -> Any:
    """Get or create a ChromaDB client for the specified path."""
    path = db_path
    if path not in _chroma_clients:
        _chroma_clients[path] = chromadb.PersistentClient(path=path)
    return _chroma_clients[path]


# ============================================================================
# Helper Functions
# ============================================================================


def get_or_create_collection(name: str, db_path: str):
    """Get or create a ChromaDB collection by name.

    Args:
        name: Name of the collection
        db_path: Path to ChromaDB database

    Returns:
        ChromaDB Collection object
    """
    client = get_chroma_client(db_path)
    return client.get_or_create_collection(name=name)


async def get_embedding(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    input_type: str = "passage",
    truncate: str = "NONE",
) -> List[float]:
    """Get embedding vector from NVIDIA NIM API for a single text.

    Args:
        text: The text to embed
        model: The embedding model to use
        input_type: 'passage' for indexing, 'query' for searching
        truncate: How to handle long inputs

    Returns:
        List of floats representing the embedding vector
    """
    embeddings = await get_embeddings_batch(
        [text], model=model, input_type=input_type, truncate=truncate
    )
    return embeddings[0]


async def get_embeddings_batch(
    texts: List[str],
    model: str = DEFAULT_EMBED_MODEL,
    input_type: str = "passage",
    truncate: str = "NONE",
) -> List[List[float]]:
    """Get embedding vectors from NVIDIA NIM API for a batch of texts.

    NVIDIA NIM supports batch inputs, which is much more efficient than
    making individual requests.

    Args:
        texts: List of texts to embed
        model: The embedding model to use
        input_type: 'passage' for indexing, 'query' for searching
        truncate: How to handle long inputs

    Returns:
        List of embedding vectors (one per input text)
    """
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


async def rerank(
    query: str,
    documents: List[str],
    model: str = DEFAULT_RERANK_MODEL,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Rerank documents by relevance to query.

    Args:
        query: The search query
        documents: List of documents to rerank
        model: The reranking model to use
        top_n: Number of top results to return

    Returns:
        List of reranked documents with scores
    """
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


def generate_document_id(text: str, filepath: str = "") -> str:
    """Generate a unique ID for a document based on its content and filepath.

    Including the filepath prevents hash collisions when different files
    contain identical code snippets (e.g., `def __init__(self): pass`).
    """
    content_to_hash = f"{filepath}::{text}" if filepath else text
    return hashlib.sha256(content_to_hash.encode()).hexdigest()[:16]


def detect_language_from_extension(file_path: str) -> str:
    """Detect programming language from file extension."""
    _, ext = os.path.splitext(file_path)
    return EXTENSION_TO_LANGUAGE.get(ext, "unknown")


async def normalize_file_content(file_path: str) -> str:
    """Read file content with proper encoding detection.

    Wrapped in asyncio.to_thread to prevent blocking the event loop.

    Args:
        file_path: Path to the file to read

    Returns:
        Content of the file as a string
    """

    def _read_file():
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "utf-16", "ascii"]

        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode file {file_path} with any tried encoding")

    return await asyncio.to_thread(_read_file)


async def _delete_entries_by_filepath(filepath: str, coll) -> int:
    """Delete all indexed entries for a given filepath from a collection.

    Args:
        filepath: The file path to remove entries for
        coll: The ChromaDB collection instance

    Returns:
        Number of entries deleted
    """
    return await _delete_entries_by_filepaths([filepath], coll)


async def _delete_entries_by_filepaths(filepaths: List[str], coll) -> int:
    """Delete all indexed entries for a list of filepaths from a collection.

    Wrapped in asyncio.to_thread to prevent blocking the event loop.

    Args:
        filepaths: List of file paths to remove entries for
        coll: The ChromaDB collection instance

    Returns:
        Number of entries deleted
    """
    if not filepaths:
        return 0

    abs_paths = [os.path.abspath(fp) for fp in filepaths]

    def _delete():
        # Use $in operator for bulk deletion if more than one path
        if len(abs_paths) > 1:
            result = coll.delete(where={"filepath": {"$in": abs_paths}})
        else:
            result = coll.delete(where={"filepath": {"$eq": abs_paths[0]}})

        return result.get("deleted", 0) if isinstance(result, dict) else 0

    return await asyncio.to_thread(_delete)


# ============================================================================
# MCP Tools
# ============================================================================


# DEPRECATED: Use index_file_by_path instead with code_content parameter
async def index_code(
    code: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: str = "default",
    model: str = DEFAULT_EMBED_MODEL,
    use_ast_chunking: bool = True,
) -> Dict[str, Any]:
    """Index a code snippet with its embedding.

    When use_ast_chunking is True and the code appears to be from a supported file type,
    the code will be split into semantic chunks (functions, classes, etc.) using AST parsing.

    Args:
        code: The code snippet to index
        metadata: Optional metadata (e.g., filename, language, description)
        collection: Name of the collection to store in
        model: Embedding model to use
        use_ast_chunking: Whether to use AST-based semantic chunking (default: True)

    Returns:
        Dict with document ID and status
    """
    try:
        coll = get_or_create_collection(collection, CHROMA_PERSIST_DIR)

        # Delete any previous entries for this filepath before re-indexing (threaded)
        if metadata and metadata.get("filepath"):
            await _delete_entries_by_filepath(metadata["filepath"], coll)

        # Check if we should use AST chunking
        if use_ast_chunking and ENABLE_AST_CHUNKING:
            # Try to get filepath from metadata for extension detection
            filepath = metadata.get("filepath", "") if metadata else ""
            if filepath and is_extension_supported(filepath):
                # Use AST-based chunking
                raw_chunks = chunk_code_by_ast(filepath, code)
                formatted_chunks = format_for_nvidia_nim(filepath, raw_chunks)

                # Process chunks in batches with concurrent embedding
                BATCH_SIZE = 15
                doc_ids = []
                all_ids = []
                all_embeddings = []
                all_documents = []
                all_metadatas = []

                for batch_start in range(0, len(formatted_chunks), BATCH_SIZE):
                    batch_chunks = formatted_chunks[
                        batch_start : batch_start + BATCH_SIZE
                    ]
                    batch_texts = [chunk["text"] for chunk in batch_chunks]

                    # Concurrent batch embedding
                    batch_embeddings = await get_embeddings_batch(
                        batch_texts, model=model, input_type="passage"
                    )

                    for i, chunk in enumerate(batch_chunks):
                        global_idx = batch_start + i
                        chunk_metadata = metadata.copy() if metadata else {}
                        chunk_metadata["chunk_index"] = global_idx
                        chunk_metadata["total_chunks"] = len(formatted_chunks)
                        chunk_metadata["chunk_type"] = chunk["type"]
                        chunk_metadata["chunk_start_line"] = chunk["start_line"]
                        chunk_metadata["chunk_end_line"] = chunk["end_line"]
                        chunk_metadata["indexed_at"] = datetime.now().isoformat()
                        chunk_metadata["model"] = model
                        chunk_metadata["language"] = detect_language_from_extension(
                            filepath
                        )

                        # FIX: Include filepath in document ID
                        chunk_doc_id = generate_document_id(
                            chunk["text"], filepath=filepath
                        )

                        all_ids.append(chunk_doc_id)
                        all_embeddings.append(batch_embeddings[i])
                        all_documents.append(chunk["text"])
                        all_metadatas.append(chunk_metadata)
                        doc_ids.append(chunk_doc_id)

                # Single batched upsert to ChromaDB (threaded)
                if all_ids:

                    def _upsert():
                        coll.upsert(
                            ids=all_ids,
                            embeddings=all_embeddings,
                            documents=all_documents,
                            metadatas=all_metadatas,
                        )

                    await asyncio.to_thread(_upsert)

                return {
                    "success": True,
                    "document_ids": doc_ids,
                    "collection": collection,
                    "message": f"Successfully indexed {len(doc_ids)} semantic chunks",
                    "chunking_method": "ast",
                    "language": detect_language_from_extension(filepath),
                }

        # Fallback to simple indexing (no chunking)
        embedding = await get_embedding(code, model=model, input_type="passage")
        # FIX: Include filepath in document ID if available
        filepath = metadata.get("filepath", "") if metadata else ""
        doc_id = generate_document_id(code, filepath=filepath)

        if metadata is None:
            metadata = {}
        metadata["indexed_at"] = datetime.now().isoformat()
        metadata["model"] = model
        metadata["language"] = metadata.get(
            "language", detect_language_from_extension(metadata.get("filename", ""))
        )

        # Threaded upsert
        def _upsert():
            coll.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[code],
                metadatas=[metadata],
            )

        await asyncio.to_thread(_upsert)

        return {
            "success": True,
            "document_id": doc_id,
            "collection": collection,
            "message": "Successfully indexed code snippet",
            "chunking_method": "none",
            "language": metadata.get("language"),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to index code: {str(e)}"}


@mcp.tool()
async def search_code(
    query: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
    limit: int = 10,
    model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    include_reranking: bool = True,
) -> Dict[str, Any]:
    """Search for code snippets using semantic similarity.

    This tool searches within the context of a project database. Each project should have its own isolated database directory.

    Args:
        query: The search query
        collection: Name of the collection to search in
        limit: Maximum number of results to return
        model: Embedding model to use for search
        rerank_model: Reranking model to use
        include_reranking: Whether to include reranking (recommended)
        db_path: Path to ChromaDB database for the project.

    Returns:
        Dict with search results and metadata
    """
    try:
        # Use the provided database path directly

        # First, get embeddings for the query
        query_embedding = await get_embedding(query, model=model, input_type="query")

        # Get collection and query ChromaDB (threaded to prevent blocking)
        coll = get_or_create_collection(collection, db_path)

        def _query_chroma():
            return coll.query(
                query_embeddings=[query_embedding],
                n_results=max(5, limit * 2),  # Get more results to rerank
                include=["documents", "metadatas", "distances"],
            )

        results = await asyncio.to_thread(_query_chroma)

        # Extract data from ChromaDB response with null safety
        docs_list = results.get("documents", [[]])
        docs_list = docs_list[0] if docs_list else []

        metas_list = results.get("metadatas", [[]])
        metas_list = metas_list[0] if metas_list else []

        dists_list = results.get("distances", [[]])
        dists_list = dists_list[0] if dists_list else []

        ids_list = results.get("ids", [[]])
        ids_list = ids_list[0] if ids_list else []

        if not docs_list:
            return {
                "success": True,
                "query": query,
                "collection": collection,
                "limit": limit,
                "count": 0,
                "results": [],
                "message": "No results found",
            }

        # Combine with IDs
        combined_results = [
            {
                "document_id": doc_id,
                "code": doc,
                "metadata": metadata or {},
                "distance": distance,
            }
            for doc_id, doc, metadata, distance in zip(
                ids_list, docs_list, metas_list, dists_list
            )
        ]

        # Rerank if requested
        if include_reranking and combined_results:
            rerank_input = [r["code"] for r in combined_results]
            rerank_results = await rerank(
                query=query,
                documents=rerank_input,
                model=rerank_model,
                top_n=limit,
            )

            # Map rerank results back to original results using enumerate
            # FIX: Previously used combined_results.index(result) which is O(N)
            # and breaks when duplicate texts exist
            rank_map = {item["index"]: item["logit"] for item in rerank_results}
            for idx, result in enumerate(combined_results):
                result["rerank_score"] = rank_map.get(idx, 0.0)

            # Sort by rerank score
            combined_results.sort(
                key=lambda x: x.get("rerank_score", 0.0),
                reverse=True,
            )
        else:
            # Sort by distance (similarity)
            combined_results.sort(key=lambda x: x.get("distance", float("inf")))

        # Return top results
        return {
            "success": True,
            "query": query,
            "collection": collection,
            "db_path": db_path,
            "limit": limit,
            "count": len(combined_results[:limit]),
            "results": combined_results[:limit],
        }
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}


async def delete_document(
    document_id: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
) -> Dict[str, Any]:
    """Delete a document from the collection.

    Args:
        document_id: The ID of the document to delete
        db_path: Path to ChromaDB database for the project.
        collection: Name of the collection

    Returns:
        Dict with deletion status
    """
    try:
        coll = get_or_create_collection(collection, db_path)

        def _delete():
            coll.delete(ids=[document_id])

        await asyncio.to_thread(_delete)
        return {
            "success": True,
            "document_id": document_id,
            "collection": collection,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to delete document: {str(e)}"}


async def delete_collection(
    collection_name: str,
    db_path: str = CHROMA_PERSIST_DIR,
) -> Dict[str, Any]:
    """Delete a collection and all its documents.

    Args:
        collection_name: Name of the collection to delete
        db_path: Custom path for ChromaDB database. If not provided, uses global.

    Returns:
        Dict with deletion status
    """
    try:
        client = get_chroma_client(db_path)

        def _delete():
            client.delete_collection(name=collection_name)

        await asyncio.to_thread(_delete)
        return {"success": True, "collection_name": collection_name}
    except Exception as e:
        return {"success": False, "error": f"Failed to delete collection: {str(e)}"}


async def list_collections(
    db_path: str = CHROMA_PERSIST_DIR,
) -> Dict[str, Any]:
    """List all available collections.

    Args:
        db_path: Custom path for ChromaDB database. If not provided, uses global.

    Returns:
        Dict with list of collection names
    """
    try:
        client = get_chroma_client(db_path)

        def _list():
            return client.list_collections()

        collections = await asyncio.to_thread(_list)
        return {
            "success": True,
            "collections": [coll.name for coll in collections],
            "count": len(collections),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to list collections: {str(e)}"}


async def create_collection(
    collection_name: str,
    db_path: str = CHROMA_PERSIST_DIR,
) -> Dict[str, Any]:
    """Create a new collection.

    Args:
        collection_name: Name for the new collection
        db_path: Custom path for ChromaDB database. If not provided, uses global.

    Returns:
        Dict with creation status
    """
    try:
        get_or_create_collection(collection_name, db_path)
        return {"success": True, "collection_name": collection_name}
    except Exception as e:
        return {"success": False, "error": f"Failed to create collection: {str(e)}"}


async def get_collection_stats(
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
) -> Dict[str, Any]:
    """Get collection statistics.

    Args:
        collection: Name of the collection
        db_path: Custom path for ChromaDB database. If not provided, will auto-detect
            project root and use .chroma_db in the project directory.
    Returns:
        Dict with collection statistics
    """
    try:
        coll = get_or_create_collection(collection, db_path)

        def _get_stats():
            stats = {
                "collection_name": collection,
                "document_count": coll.count(),
            }
            # Get sample documents
            sample_results = coll.get(limit=5)
            if sample_results["documents"]:
                stats["sample_documents"] = [
                    {"document_id": doc_id, "content": (doc[:100] if doc else "")}
                    for doc_id, doc in zip(
                        sample_results["ids"], sample_results["documents"]
                    )
                ]
            return stats

        stats = await asyncio.to_thread(_get_stats)
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": f"Failed to get collection stats: {str(e)}"}


# ============================================================================
# Deprecated Tools - These are kept as internal helpers but are DEPRECATED
# Use index_file_by_path with code_content parameter instead
# ============================================================================
async def batch_index_codes(
    codes: List[str],
    db_path: str = CHROMA_PERSIST_DIR,
    base_metadata: Optional[Dict[str, Any]] = None,
    collection: str = "default",
    model: str = DEFAULT_EMBED_MODEL,
) -> Dict[str, Any]:
    """DEPRECATED: Index multiple code snippets in batch.

    Use index_file_by_path with code_content parameter instead.

    Args:
        codes: List of code snippets to index
        db_path: Path to ChromaDB database for the project.
        base_metadata: Optional base metadata to apply to all
        collection: Name of the collection to store in
        model: Embedding model to use

    Returns:
        Dict with batch indexing results
    """
    try:
        client = get_chroma_client(db_path)
        coll = client.get_or_create_collection(name=collection)

        embeddings = []
        doc_ids = []
        metadatas = []
        successful = 0
        failed = 0

        # Use batch embedding API for efficiency
        try:
            batch_embeddings = await get_embeddings_batch(
                codes, model=model, input_type="passage"
            )
            for i, code in enumerate(codes):
                try:
                    embedding = batch_embeddings[i]
                    doc_id = generate_document_id(code)
                    metadata = base_metadata.copy() if base_metadata else {}
                    metadata["indexed_at"] = datetime.now().isoformat()
                    metadata["model"] = model
                    embeddings.append(embedding)
                    doc_ids.append(doc_id)
                    metadatas.append(metadata)
                    successful += 1
                except Exception:
                    failed += 1
        except Exception:
            # Fallback to individual embeddings if batch fails
            for code in codes:
                try:
                    embedding = await get_embedding(
                        code, model=model, input_type="passage"
                    )
                    doc_id = generate_document_id(code)
                    metadata = base_metadata.copy() if base_metadata else {}
                    metadata["indexed_at"] = datetime.now().isoformat()
                    metadata["model"] = model
                    embeddings.append(embedding)
                    doc_ids.append(doc_id)
                    metadatas.append(metadata)
                    successful += 1
                except Exception:
                    failed += 1

        # Batch insert into ChromaDB (threaded)
        if embeddings:

            def _upsert():
                coll.upsert(
                    ids=doc_ids,
                    embeddings=embeddings,
                    documents=codes[:successful],
                    metadatas=metadatas,
                )

            await asyncio.to_thread(_upsert)

        return {
            "success": True,
            "total": len(codes),
            "successful": successful,
            "failed": failed,
            "document_ids": doc_ids,
            "collection": collection,
        }
    except Exception as e:
        return {"success": False, "error": f"Batch indexing failed: {str(e)}"}


async def health_check() -> Dict[str, Any]:
    """Check server health and API connectivity.

    Returns:
        Dict with health status
    """
    status = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(NVIDIA_API_KEY),
        "models": {
            "embed_model": DEFAULT_EMBED_MODEL,
            "rerank_model": DEFAULT_RERANK_MODEL,
        },
    }

    # Test API connectivity
    if NVIDIA_API_KEY:
        test_text = "test"
        try:
            await get_embedding(test_text, input_type="query")
            status["api_connected"] = True
        except Exception:
            status["api_connected"] = False
    else:
        status["api_connected"] = False

    # Test ChromaDB connectivity (threaded)
    try:
        test_coll = get_or_create_collection("health_check", CHROMA_PERSIST_DIR)

        def _count():
            return test_coll.count()

        await asyncio.to_thread(_count)
        status["chroma_connected"] = True
    except Exception:
        status["chroma_connected"] = False

    status["healthy"] = (
        status.get("api_key_configured", False)
        and status.get("api_connected", False)
        and status.get("chroma_connected", False)
    )

    return status


def get_supported_languages() -> Dict[str, Any]:
    """Get list of supported programming languages and their file extensions.

    Returns:
        Dict with supported languages and extensions mapping
    """
    try:
        return {
            "success": True,
            "languages": SUPPORTED_EXTENSIONS,
            "total_languages": len(SUPPORTED_EXTENSIONS),
            "file_extensions": list(EXTENSION_TO_LANGUAGE.keys()),
            "max_file_size": MAX_FILE_SIZE,
            "ast_chunking_enabled": ENABLE_AST_CHUNKING,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get supported languages: {str(e)}",
        }


def get_ast_chunking_info() -> Dict[str, Any]:
    """Get information about AST-based chunking support.

    Returns details about which file extensions support AST-based semantic chunking,
    including the node types that will be extracted for each language.

    Returns:
        Dict with AST chunking configuration and supported extensions
    """
    try:
        return {
            "success": True,
            "enabled": ENABLE_AST_CHUNKING,
            "max_chunk_size": MAX_CHUNK_SIZE,
            "supported_extensions": get_supported_extensions(),
            "description": "AST chunking splits code into semantic units (functions, classes, etc.) for better embedding and retrieval",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get AST chunking info: {str(e)}",
        }


async def index_file_by_path(
    filepath: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
    model: str = DEFAULT_EMBED_MODEL,
    use_content_as_document: bool = True,
    custom_metadata: Optional[Dict[str, Any]] = None,
    use_ast_chunking: bool = True,
    code_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Index a file by its path.

    PRIMARY USE: Index files from disk. Use 'filepath' to specify which file to index.
    The file content will be read automatically from disk.

    FALLBACK USE (code_content): Only use 'code_content' when you need to index code
    that does NOT exist as a file on disk (e.g., generated code, clipboard content,
    or code from an external source). When 'code_content' is provided, it is treated
    as if it were the content of the file at 'filepath', but 'filepath' must still
    be provided for metadata purposes.

    When use_ast_chunking is True and the file extension is supported, the file will
    be split into semantic chunks (functions, classes, etc.) using AST parsing before
    indexing.

    Args:
        filepath: Absolute path to the file to index (MUST BE PROVIDED - used for metadata)
        collection: Name of the collection to store in
        model: Embedding model to use
        use_content_as_document: Whether to use file content as document (True)
            or just store metadata (False)
        custom_metadata: Additional metadata to attach to the file
        use_ast_chunking: Whether to use AST-based semantic chunking (default: True)
        code_content: FALLBACK ONLY - Provide code directly instead of reading from file.
            Only use when the code does not exist on disk!
        db_path: Path to ChromaDB database for the project.

    Returns:
        Dict with indexing status and metadata
    """
    try:
        start_time = time.time()

        # Determine file content source
        if code_content is not None:
            # FALLBACK: Use provided code content instead of reading file
            file_content = code_content
            file_size = len(code_content.encode("utf-8"))
        else:
            # PRIMARY: Read file from disk
            # Check if file exists
            if not os.path.exists(filepath):
                return {"success": False, "error": "File not found"}

            # Check if it's a file
            if not os.path.isfile(filepath):
                return {"success": False, "error": "Path is not a file"}

            # Check if file is too large
            file_size = os.path.getsize(filepath)
            if file_size > MAX_FILE_SIZE:
                return {"success": False, "error": "File too large"}

            # Read file content (threaded to prevent blocking)
            file_content = await normalize_file_content(filepath)

        # Extract metadata
        file_metadata = custom_metadata.copy() if custom_metadata else {}
        file_metadata["filepath"] = os.path.abspath(filepath)
        file_metadata["filename"] = os.path.basename(filepath)
        file_metadata["fileext"] = os.path.splitext(filepath)[1]
        file_metadata["file_size"] = file_size
        file_metadata["language"] = detect_language_from_extension(filepath)

        # Handle file content vs metadata-only indexing
        if use_content_as_document:
            coll = get_or_create_collection(collection, db_path)

            # Delete any previous entries for this file before re-indexing (threaded)
            await _delete_entries_by_filepath(filepath, coll)

            # Check if we should use AST chunking
            if (
                use_ast_chunking
                and ENABLE_AST_CHUNKING
                and is_extension_supported(filepath)
            ):
                # Use AST-based chunking
                raw_chunks = chunk_code_by_ast(filepath, file_content)
                formatted_chunks = format_for_nvidia_nim(filepath, raw_chunks)

                # Process chunks in batches with concurrent embedding requests
                BATCH_SIZE = 15
                doc_ids = []
                all_ids = []
                all_embeddings = []
                all_documents = []
                all_metadatas = []

                for batch_start in range(0, len(formatted_chunks), BATCH_SIZE):
                    batch_chunks = formatted_chunks[
                        batch_start : batch_start + BATCH_SIZE
                    ]
                    batch_texts = [chunk["text"] for chunk in batch_chunks]

                    # Concurrent batch embedding
                    batch_embeddings = await get_embeddings_batch(
                        batch_texts, model=model, input_type="passage"
                    )

                    for i, chunk in enumerate(batch_chunks):
                        global_idx = batch_start + i
                        chunk_metadata = file_metadata.copy()
                        chunk_metadata["chunk_index"] = global_idx
                        chunk_metadata["total_chunks"] = len(formatted_chunks)
                        chunk_metadata["chunk_type"] = chunk["type"]
                        chunk_metadata["chunk_start_line"] = chunk["start_line"]
                        chunk_metadata["chunk_end_line"] = chunk["end_line"]
                        chunk_metadata["indexed_at"] = datetime.now().isoformat()
                        chunk_metadata["model"] = model
                        chunk_metadata["method"] = "index_file_by_path_ast_chunk"

                        # FIX: Include filepath in document ID to prevent hash collisions
                        chunk_doc_id = generate_document_id(
                            chunk["text"], filepath=filepath
                        )

                        all_ids.append(chunk_doc_id)
                        all_embeddings.append(batch_embeddings[i])
                        all_documents.append(chunk["text"])
                        all_metadatas.append(chunk_metadata)
                        doc_ids.append(chunk_doc_id)

                # Single batched upsert to ChromaDB (threaded)
                if all_ids:

                    def _upsert():
                        coll.upsert(
                            ids=all_ids,
                            embeddings=all_embeddings,
                            documents=all_documents,
                            metadatas=all_metadatas,
                        )

                    await asyncio.to_thread(_upsert)

                return {
                    "success": True,
                    "document_ids": doc_ids,
                    "filepath": os.path.abspath(filepath),
                    "language": file_metadata["language"],
                    "collection": collection,
                    "db_path": db_path,
                    "message": f"Successfully indexed {len(doc_ids)} semantic chunks",
                    "chunking_method": "ast",
                    "metadata": file_metadata,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                }

            # Fallback to simple indexing (no chunking)
            document_to_index = file_content
            embedding = await get_embedding(
                document_to_index, model=model, input_type="passage"
            )
            # FIX: Include filepath in document ID
            doc_id = generate_document_id(file_content, filepath=filepath)

            # Add indexing metadata
            file_metadata["indexed_at"] = datetime.now().isoformat()
            file_metadata["model"] = model
            file_metadata["method"] = "index_file_by_path"
            file_metadata["original_content_length"] = len(file_content)

            # Threaded upsert
            def _upsert():
                coll.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[document_to_index],
                    metadatas=[file_metadata],
                )

            await asyncio.to_thread(_upsert)

            return {
                "success": True,
                "document_id": doc_id,
                "filepath": os.path.abspath(filepath),
                "language": file_metadata["language"],
                "collection": collection,
                "db_path": db_path,
                "message": "Successfully indexed file by path",
                "chunking_method": "none",
                "metadata": file_metadata,
                "elapsed_seconds": round(time.time() - start_time, 2),
            }
        else:
            # Only index metadata, not content
            coll = get_or_create_collection(collection, db_path)

            # Delete any previous entries for this file before re-indexing (threaded)
            await _delete_entries_by_filepath(filepath, coll)

            # FIX: Include filepath in document ID
            doc_id = generate_document_id(
                json.dumps(file_metadata, sort_keys=True), filepath=filepath
            )

            # Add empty document (threaded)
            def _upsert():
                coll.upsert(
                    ids=[doc_id],
                    embeddings=[],  # No embedding for metadata-only documents
                    documents=[""],  # Empty document
                    metadatas=[file_metadata],
                )

            await asyncio.to_thread(_upsert)

            return {
                "success": True,
                "document_id": doc_id,
                "filepath": os.path.abspath(filepath),
                "language": file_metadata["language"],
                "collection": collection,
                "db_path": db_path,
                "method": "metadata_only",
                "message": "Successfully indexed file metadata only",
                "elapsed_seconds": round(time.time() - start_time, 2),
            }
    except Exception as e:
        import traceback

        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return {"success": False, "error": f"Failed to index file: {error_details}"}


async def index_directory(
    directory_path: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
    model: str = DEFAULT_EMBED_MODEL,
    extensions: Optional[List[str]] = None,
    skip_dirs: Optional[List[str]] = None,
    max_file_size: int = MAX_FILE_SIZE,
    batch_size: int = 10,
    use_ast_chunking: bool = True,
) -> Dict[str, Any]:
    """Index an entire directory tree for code files.

    The database will be automatically created in a .chroma_db folder within
    the detected project root (based on marker files like .git, package.json, etc.).
    Each project gets its own isolated database directory.

    When use_ast_chunking is True and files have supported extensions, files will
    be split into semantic chunks (functions, classes, etc.) using AST parsing before
    indexing.

    Args:
        directory_path: Absolute path to the directory to index
        collection: Name of the collection to store in
        model: Embedding model to use
        extensions: List of file extensions to include (None for all supported)
        skip_dirs: List of directory names to skip
        max_file_size: Maximum file size in bytes to index
        batch_size: Batch size for efficient processing
        use_ast_chunking: Whether to use AST-based semantic chunking (default: True)
        db_path: Path to ChromaDB database for the project.

    Returns:
        Dict with indexing statistics
    """
    start_time = time.time()

    try:
        # Validate directory
        if not os.path.exists(directory_path):
            return {"success": False, "error": "Directory not found"}

        if not os.path.isdir(directory_path):
            return {"success": False, "error": "Path is not a directory"}

        # Set defaults
        if extensions is None:
            extensions = list(EXTENSION_TO_LANGUAGE.keys())

        if skip_dirs is None:
            skip_dirs = list(SKIP_DIRS)
            
        skipped_files = 0
        # Find all code files recursively
        code_files = []
        for root, dirs, files in os.walk(directory_path, topdown=True):
            # Remove skip directories from traversal
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                file_path = os.path.join(root, file)

                # Skip based on file extension (if provided)
                _, ext = os.path.splitext(file)
                if ext not in extensions:
                    skipped_files += 1
                    continue

                # Skip if file is too large
                if os.path.getsize(file_path) > max_file_size:
                    skipped_files += 1
                    continue

                code_files.append(file_path)

        if not code_files:
            # Still clean up orphaned entries for this directory
            coll = get_or_create_collection(collection, db_path)
            abs_dir_path = os.path.abspath(directory_path) + os.sep

            def _get_metadatas():
                return coll.get(include=["metadatas"])["metadatas"] or []

            existing_metadatas = await asyncio.to_thread(_get_metadatas)

            tracked_files_in_dir = {
                m["filepath"]
                for m in existing_metadatas
                if m
                and "filepath" in m
                and isinstance(m["filepath"], str)
                and m["filepath"].startswith(abs_dir_path)
            }
            removed_orphans = len(tracked_files_in_dir)
            await _delete_entries_by_filepaths(list(tracked_files_in_dir), coll)

            return {
                "success": True,
                "directory": os.path.abspath(directory_path),
                "collection": collection,
                "db_path": db_path,
                "total_files": 0,
                "indexed_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "removed_orphans": removed_orphans,
                "document_ids": [],
                "message": "No files found to index",
                "elapsed_seconds": round(time.time() - start_time, 2),
            }

        # Process files in batches
        total_files = len(code_files)
        indexed_files = 0
        failed_files = 0
        document_ids = []

        coll = get_or_create_collection(collection, db_path)

        # Get existing tracked filepaths in this directory for orphan cleanup (threaded)
        abs_dir_path = os.path.abspath(directory_path) + os.sep

        def _get_metadatas():
            return coll.get(include=["metadatas"])["metadatas"] or []

        existing_metadatas = await asyncio.to_thread(_get_metadatas)

        tracked_files_in_dir = {
            m["filepath"]
            for m in existing_metadatas
            if m
            and "filepath" in m
            and isinstance(m["filepath"], str)
            and m["filepath"].startswith(abs_dir_path)
        }
        successfully_indexed_filepaths: set[str] = set()

        # Helper to process a single file
        async def _process_file(file_path: str):
            nonlocal indexed_files, failed_files

            try:
                # Read file content (threaded)
                file_content = await normalize_file_content(file_path)

                # Create base metadata
                base_metadata = {
                    "filepath": os.path.abspath(file_path),
                    "filename": os.path.basename(file_path),
                    "language": detect_language_from_extension(file_path),
                    "fileext": os.path.splitext(file_path)[1],
                    "file_size": os.path.getsize(file_path),
                    "indexed_at": datetime.now().isoformat(),
                    "model": model,
                    "method": "index_directory",
                }

                # Delete any previous entries for this file before re-indexing (threaded)
                await _delete_entries_by_filepath(file_path, coll)

                # Check if we should use AST chunking
                if (
                    use_ast_chunking
                    and ENABLE_AST_CHUNKING
                    and is_extension_supported(file_path)
                ):
                    # Use AST-based chunking
                    raw_chunks = chunk_code_by_ast(file_path, file_content)
                    formatted_chunks = format_for_nvidia_nim(file_path, raw_chunks)

                    # Process chunks in batches with concurrent embedding
                    CHUNK_BATCH_SIZE = 15
                    file_doc_ids = []
                    all_ids = []
                    all_embeddings = []
                    all_documents = []
                    all_metadatas = []

                    for batch_start in range(
                        0, len(formatted_chunks), CHUNK_BATCH_SIZE
                    ):
                        batch_chunks = formatted_chunks[
                            batch_start : batch_start + CHUNK_BATCH_SIZE
                        ]
                        batch_texts = [chunk["text"] for chunk in batch_chunks]

                        # Concurrent batch embedding
                        batch_embeddings = await get_embeddings_batch(
                            batch_texts, model=model, input_type="passage"
                        )

                        for i, chunk in enumerate(batch_chunks):
                            global_idx = batch_start + i
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata["chunk_index"] = global_idx
                            chunk_metadata["total_chunks"] = len(formatted_chunks)
                            chunk_metadata["chunk_type"] = chunk["type"]
                            chunk_metadata["chunk_start_line"] = chunk["start_line"]
                            chunk_metadata["chunk_end_line"] = chunk["end_line"]

                            # FIX: Include filepath in document ID
                            chunk_doc_id = generate_document_id(
                                chunk["text"], filepath=file_path
                            )

                            all_ids.append(chunk_doc_id)
                            all_embeddings.append(batch_embeddings[i])
                            all_documents.append(chunk["text"])
                            all_metadatas.append(chunk_metadata)
                            file_doc_ids.append(chunk_doc_id)

                    # Single batched upsert to ChromaDB (threaded)
                    if all_ids:

                        def _upsert():
                            coll.upsert(
                                ids=all_ids,
                                embeddings=all_embeddings,
                                documents=all_documents,
                                metadatas=all_metadatas,
                            )

                        await asyncio.to_thread(_upsert)

                    return file_doc_ids
                else:
                    # Simple indexing without AST chunking
                    embedding = await get_embedding(
                        file_content, model=model, input_type="passage"
                    )
                    # FIX: Include filepath in document ID
                    doc_id = generate_document_id(file_content, filepath=file_path)

                    # Threaded upsert
                    def _upsert():
                        coll.upsert(
                            ids=[doc_id],
                            embeddings=[embedding],
                            documents=[file_content],
                            metadatas=[base_metadata],
                        )

                    await asyncio.to_thread(_upsert)

                    return [doc_id]

            except Exception:
                failed_files += 1
                return []

        # Process files concurrently in batches bounded by semaphore
        FILE_BATCH_SIZE = 5
        for batch_start in range(0, len(code_files), FILE_BATCH_SIZE):
            batch_files = code_files[batch_start : batch_start + FILE_BATCH_SIZE]

            # Create tasks for concurrent processing
            tasks = [_process_file(fp) for fp in batch_files]
            batch_results = await asyncio.gather(*tasks)

            for i, file_doc_ids in enumerate(batch_results):
                if file_doc_ids:
                    indexed_files += 1
                    document_ids.extend(file_doc_ids)
                    # Track successfully indexed filepaths using enumerate index
                    file_idx = batch_start + i
                    if file_idx < len(code_files):
                        successfully_indexed_filepaths.add(
                            os.path.abspath(code_files[file_idx])
                        )

        # Remove orphaned entries for files that no longer exist in this directory (threaded)
        orphaned_files = tracked_files_in_dir - successfully_indexed_filepaths
        removed_orphans = len(orphaned_files)
        await _delete_entries_by_filepaths(list(orphaned_files), coll)

        return {
            "success": True,
            "directory": os.path.abspath(directory_path),
            "collection": collection,
            "db_path": db_path,
            "total_files": total_files,
            "indexed_files": indexed_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
            "removed_orphans": removed_orphans,
            "document_ids": document_ids,
            "message": "Directory indexing completed",
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
    except Exception as e:
        elapsed = round(time.time() - start_time, 2) if "start_time" in dir() else None
        return {
            "success": False,
            "error": f"Failed to index directory: {str(e)}",
            "elapsed_seconds": elapsed,
        }


# ============================================================================
# Main Entry Point
# ============================================================================


async def _cleanup_http_client():
    """Cleanup the global HTTP client on shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


def main():
    """Run the MCP server."""

    print("Starting NVIDIA NIM MCP Server...")
    print(f"ChromaDB path: {CHROMA_PERSIST_DIR}")
    print(f"Embed model: {DEFAULT_EMBED_MODEL}")
    print(f"Rerank model: {DEFAULT_RERANK_MODEL}")
    print(f"Supported languages: {len(SUPPORTED_EXTENSIONS)}")
    print()

    # Conditional tool registration based on SERVER_MODE
    # Default (unset) is "search" mode with only search_code
    SERVER_MODE = os.getenv("NIM_SERVER_MODE", "search").lower()
    print(f"Server mode: {SERVER_MODE}")

    if SERVER_MODE == "admin":
        # Admin mode: register all management and admin tools
        mcp.add_tool(delete_document)
        mcp.add_tool(delete_collection)
        mcp.add_tool(list_collections)
        mcp.add_tool(create_collection)
        mcp.add_tool(get_collection_stats)
        mcp.add_tool(health_check)
        mcp.add_tool(get_supported_languages)
        mcp.add_tool(get_ast_chunking_info)
        mcp.add_tool(index_file_by_path)
        mcp.add_tool(index_directory)
        print(
            "Registered admin tools: delete_document, delete_collection, list_collections, create_collection, get_collection_stats, health_check, get_supported_languages, get_ast_chunking_info, index_file_by_path, index_directory, search_code"
        )
    elif SERVER_MODE == "manage":
        # Manage mode: register only management tools
        mcp.add_tool(index_file_by_path)
        mcp.add_tool(index_directory)
        print(
            "Registered manage tools: index_file_by_path, index_directory, search_code"
        )
    elif SERVER_MODE == "search":
        # Search mode (default): only search_code is available via @mcp.tool() decorator
        print("Registered search tools: search_code")
    else:
        # Unknown mode: default to search mode
        print(f"Unknown mode '{SERVER_MODE}', defaulting to search mode")
        print("Registered search tools: search_code")

    print()
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
