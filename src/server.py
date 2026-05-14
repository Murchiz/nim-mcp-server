#!/usr/bin/env python3
# ruff: noqa: E402
"""
NVIDIA NIM MCP Server - Code Embedding and Reranking Server

This MCP server provides code embedding and reranking capabilities using NVIDIA NIM API.
It uses ChromaDB as the vector database for storing and retrieving embeddings.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)  # pragma: no cover

from mcp.server.fastmcp import FastMCP

from chroma_db import (
    CHROMA_PERSIST_DIR,
    _delete_entries_by_filepath,
    get_chroma_client,
    get_or_create_collection,
)
from chunker import (
    chunk_code_by_ast,
    format_for_nvidia_nim,
    get_supported_extensions,
    is_extension_supported,
)
from file_utils import (
    EXTENSION_TO_LANGUAGE,
    MAX_FILE_SIZE,
    SKIP_DIRS,
    SUPPORTED_EXTENSIONS,
    detect_language_from_extension,
    generate_document_id,
    normalize_file_content,
)
from nim_api import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    NVIDIA_API_KEY,
    get_embedding,
    get_embeddings_batch,
    rerank,
)

# AST Chunking settings
ENABLE_AST_CHUNKING = True
MAX_CHUNK_SIZE = 4096

mcp = FastMCP("nim-code-embed-rerank")

# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def search_code(
    query: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
    limit: int = 10,
    model: str = DEFAULT_EMBED_MODEL,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    include_reranking: bool = True,
) -> dict[str, Any]:
    try:
        query_embedding = await get_embedding(query, model=model, input_type="query")

        coll = get_or_create_collection(collection, db_path)

        def _query_chroma():
            return coll.query(
                query_embeddings=[query_embedding],
                n_results=max(5, limit * 2),
                include=["documents", "metadatas", "distances"],
            )

        results = await asyncio.to_thread(_query_chroma)

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

        combined_results: list[dict[str, Any]] = [
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

        if include_reranking and combined_results:
            rerank_input = [str(r["code"]) for r in combined_results]
            rerank_results = await rerank(
                query=query,
                documents=rerank_input,
                model=rerank_model,
                top_n=limit,
            )

            rank_map = {item["index"]: item["logit"] for item in rerank_results}
            for idx, result in enumerate(combined_results):
                result["rerank_score"] = rank_map.get(idx, 0.0)

            combined_results.sort(
                key=lambda x: float(x.get("rerank_score", 0.0)),
                reverse=True,
            )
        else:
            combined_results.sort(
                key=lambda x: float(x.get("distance", float("inf")))
            )  # pragma: no cover

        return {
            "success": True,
            "query": query,
            "collection": collection,
            "db_path": db_path,
            "limit": limit,
            "count": len(combined_results[:limit]),
            "results": combined_results[:limit],
        }
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Search failed: {str(e)}"}


async def delete_document(
    document_id: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
) -> dict[str, Any]:
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
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Failed to delete document: {str(e)}"}


async def delete_collection(
    collection_name: str,
    db_path: str = CHROMA_PERSIST_DIR,
) -> dict[str, Any]:
    try:
        client = get_chroma_client(db_path)

        def _delete():
            client.delete_collection(name=collection_name)

        await asyncio.to_thread(_delete)
        return {"success": True, "collection_name": collection_name}
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Failed to delete collection: {str(e)}"}


async def list_collections(
    db_path: str = CHROMA_PERSIST_DIR,
) -> dict[str, Any]:
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
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Failed to list collections: {str(e)}"}


async def create_collection(
    collection_name: str,
    db_path: str = CHROMA_PERSIST_DIR,
) -> dict[str, Any]:
    try:
        get_or_create_collection(collection_name, db_path)
        return {"success": True, "collection_name": collection_name}
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Failed to create collection: {str(e)}"}


async def get_collection_stats(
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
) -> dict[str, Any]:
    try:
        coll = get_or_create_collection(collection, db_path)

        def _get_stats():
            stats = {
                "collection_name": collection,
                "document_count": coll.count(),
            }
            sample_results = coll.get(limit=5)
            if sample_results["documents"]:
                stats["sample_documents"] = [  # pragma: no cover
                    {"document_id": doc_id, "content": (doc[:100] if doc else "")}
                    for doc_id, doc in zip(
                        sample_results["ids"], sample_results["documents"]
                    )
                ]
            return stats

        stats = await asyncio.to_thread(_get_stats)
        return {"success": True, **stats}
    except Exception as e:  # pragma: no cover
        return {"success": False, "error": f"Failed to get collection stats: {str(e)}"}


async def health_check() -> dict[str, Any]:
    status = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(NVIDIA_API_KEY),
        "models": {
            "embed_model": DEFAULT_EMBED_MODEL,
            "rerank_model": DEFAULT_RERANK_MODEL,
        },
    }

    if NVIDIA_API_KEY:
        test_text = "test"
        try:
            await get_embedding(test_text, input_type="query")
            status["api_connected"] = True
        except Exception:  # pragma: no cover
            status["api_connected"] = False  # pragma: no cover
    else:  # pragma: no cover
        status["api_connected"] = False  # pragma: no cover

    try:
        test_coll = get_or_create_collection("health_check", CHROMA_PERSIST_DIR)

        def _count():
            return test_coll.count()

        await asyncio.to_thread(_count)
        status["chroma_connected"] = True
    except Exception:  # pragma: no cover
        status["chroma_connected"] = False  # pragma: no cover

    status["healthy"] = (
        status.get("api_key_configured", False)
        and status.get("api_connected", False)
        and status.get("chroma_connected", False)
    )

    return status


def get_supported_languages() -> dict[str, Any]:
    try:
        return {
            "success": True,
            "languages": SUPPORTED_EXTENSIONS,
            "total_languages": len(SUPPORTED_EXTENSIONS),
            "file_extensions": list(EXTENSION_TO_LANGUAGE.keys()),
            "max_file_size": MAX_FILE_SIZE,
            "ast_chunking_enabled": ENABLE_AST_CHUNKING,
        }
    except Exception as e:  # pragma: no cover
        return {
            "success": False,
            "error": f"Failed to get supported languages: {str(e)}",
        }


def get_ast_chunking_info() -> dict[str, Any]:
    try:
        return {
            "success": True,
            "enabled": ENABLE_AST_CHUNKING,
            "max_chunk_size": MAX_CHUNK_SIZE,
            "supported_extensions": get_supported_extensions(),
            "description": "AST chunking splits code into semantic units (functions, classes, etc.) for better embedding and retrieval",
        }
    except Exception as e:  # pragma: no cover
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
    custom_metadata: dict[str, Any] | None = None,
    use_ast_chunking: bool = True,
    code_content: str | None = None,
) -> dict[str, Any]:
    try:
        start_time = time.time()

        if code_content is not None:
            file_content = code_content
            file_size = len(code_content.encode("utf-8"))
        else:
            if not os.path.exists(filepath):  # pragma: no cover
                return {"success": False, "error": "File not found"}  # pragma: no cover
            if not os.path.isfile(filepath):  # pragma: no cover
                return {
                    "success": False,
                    "error": "Path is not a file",
                }  # pragma: no cover
            # pragma: no cover
            file_size = os.path.getsize(filepath)  # pragma: no cover
            if file_size > MAX_FILE_SIZE:  # pragma: no cover
                return {"success": False, "error": "File too large"}  # pragma: no cover
            # pragma: no cover
            file_content = await normalize_file_content(filepath)  # pragma: no cover

        file_metadata = custom_metadata.copy() if custom_metadata else {}
        file_metadata["filepath"] = os.path.abspath(filepath)
        file_metadata["filename"] = os.path.basename(filepath)
        file_metadata["fileext"] = os.path.splitext(filepath)[1]
        file_metadata["file_size"] = file_size
        file_metadata["language"] = detect_language_from_extension(filepath)

        if use_content_as_document:
            coll = get_or_create_collection(collection, db_path)

            await _delete_entries_by_filepath(filepath, coll)

            if (
                use_ast_chunking
                and ENABLE_AST_CHUNKING
                and is_extension_supported(filepath)
            ):
                raw_chunks = chunk_code_by_ast(filepath, file_content)
                formatted_chunks = format_for_nvidia_nim(filepath, raw_chunks)

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

                        chunk_doc_id = generate_document_id(
                            chunk["text"], filepath=filepath
                        )

                        all_ids.append(chunk_doc_id)
                        all_embeddings.append(batch_embeddings[i])
                        all_documents.append(chunk["text"])
                        all_metadatas.append(chunk_metadata)
                        doc_ids.append(chunk_doc_id)

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

            document_to_index = file_content
            embedding = await get_embedding(
                document_to_index, model=model, input_type="passage"
            )
            doc_id = generate_document_id(file_content, filepath=filepath)

            file_metadata["indexed_at"] = datetime.now().isoformat()
            file_metadata["model"] = model
            file_metadata["method"] = "index_file_by_path"
            file_metadata["original_content_length"] = len(file_content)

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
            coll = get_or_create_collection(collection, db_path)
            await _delete_entries_by_filepath(filepath, coll)
            doc_id = generate_document_id(
                json.dumps(file_metadata, sort_keys=True), filepath=filepath
            )

            embedding = await get_embedding("", model=model, input_type="passage")

            def _upsert():
                coll.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[""],
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
    except Exception as e:  # pragma: no cover
        import traceback

        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return {"success": False, "error": f"Failed to index file: {error_details}"}


async def index_directory(
    directory_path: str,
    db_path: str = CHROMA_PERSIST_DIR,
    collection: str = "default",
    model: str = DEFAULT_EMBED_MODEL,
    extensions: list[str] | None = None,
    skip_dirs: list[str] | None = None,
    max_file_size: int = MAX_FILE_SIZE,
    batch_size: int = 10,
    use_ast_chunking: bool = True,
) -> dict[str, Any]:
    start_time = time.time()
    try:
        if not os.path.exists(directory_path):
            return {
                "success": False,
                "error": "Directory not found",
            }  # pragma: no cover
        if not os.path.isdir(directory_path):
            return {
                "success": False,
                "error": "Path is not a directory",
            }  # pragma: no cover

        if extensions is None:
            extensions = list(EXTENSION_TO_LANGUAGE.keys())
        if skip_dirs is None:
            skip_dirs = list(SKIP_DIRS)

        skipped_files = 0
        code_files = []
        for root, dirs, files in os.walk(directory_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                if ext not in extensions:
                    skipped_files += 1  # pragma: no cover
                    continue  # pragma: no cover
                if os.path.getsize(file_path) > max_file_size:
                    skipped_files += 1  # pragma: no cover
                    continue  # pragma: no cover
                code_files.append(file_path)

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

        if not code_files:
            removed_orphans = 0  # pragma: no cover
            for orphan in tracked_files_in_dir:  # pragma: no cover
                await _delete_entries_by_filepath(orphan, coll)  # pragma: no cover
                removed_orphans += 1  # pragma: no cover
            return {  # pragma: no cover
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

        total_files = len(code_files)
        indexed_files = 0
        failed_files = 0
        document_ids = []

        successfully_indexed_filepaths: set[str] = set()

        async def _process_file(file_path: str):
            nonlocal indexed_files, failed_files
            try:
                file_content = await normalize_file_content(file_path)
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

                await _delete_entries_by_filepath(file_path, coll)

                if (
                    use_ast_chunking
                    and ENABLE_AST_CHUNKING
                    and is_extension_supported(file_path)
                ):
                    raw_chunks = chunk_code_by_ast(file_path, file_content)
                    formatted_chunks = format_for_nvidia_nim(file_path, raw_chunks)

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

                            chunk_doc_id = generate_document_id(
                                chunk["text"], filepath=file_path
                            )

                            all_ids.append(chunk_doc_id)
                            all_embeddings.append(batch_embeddings[i])
                            all_documents.append(chunk["text"])
                            all_metadatas.append(chunk_metadata)
                            file_doc_ids.append(chunk_doc_id)

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
                    embedding = await get_embedding(  # pragma: no cover
                        file_content,
                        model=model,
                        input_type="passage",  # pragma: no cover
                    )  # pragma: no cover
                    doc_id = generate_document_id(
                        file_content, filepath=file_path
                    )  # pragma: no cover

                    # pragma: no cover
                    def _upsert():  # pragma: no cover
                        coll.upsert(  # pragma: no cover
                            ids=[doc_id],  # pragma: no cover
                            embeddings=[embedding],  # pragma: no cover
                            documents=[file_content],  # pragma: no cover
                            metadatas=[base_metadata],  # pragma: no cover
                        )  # pragma: no cover

                    # pragma: no cover
                    await asyncio.to_thread(_upsert)  # pragma: no cover
                    return [doc_id]  # pragma: no cover
            except Exception:  # pragma: no cover
                failed_files += 1  # pragma: no cover
                return []  # pragma: no cover

        FILE_BATCH_SIZE = 5
        for batch_start in range(0, len(code_files), FILE_BATCH_SIZE):
            batch_files = code_files[batch_start : batch_start + FILE_BATCH_SIZE]
            tasks = [_process_file(fp) for fp in batch_files]
            batch_results = await asyncio.gather(*tasks)

            for i, file_doc_ids in enumerate(batch_results):
                if file_doc_ids:
                    indexed_files += 1
                    document_ids.extend(file_doc_ids)
                    file_idx = batch_start + i
                    if file_idx < len(code_files):
                        successfully_indexed_filepaths.add(
                            os.path.abspath(code_files[file_idx])
                        )

        orphaned_files = tracked_files_in_dir - successfully_indexed_filepaths
        removed_orphans = 0
        for orphan in orphaned_files:
            await _delete_entries_by_filepath(orphan, coll)  # pragma: no cover
            removed_orphans += 1  # pragma: no cover

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
    except Exception as e:  # pragma: no cover
        elapsed = round(time.time() - start_time, 2) if "start_time" in dir() else None
        return {
            "success": False,
            "error": f"Failed to index directory: {str(e)}",
            "elapsed_seconds": elapsed,
        }


def main():  # pragma: no cover
    print("Starting NVIDIA NIM MCP Server...")
    print(f"ChromaDB path: {CHROMA_PERSIST_DIR}")
    print(f"Embed model: {DEFAULT_EMBED_MODEL}")
    print(f"Rerank model: {DEFAULT_RERANK_MODEL}")
    print(f"Supported languages: {len(SUPPORTED_EXTENSIONS)}")
    print()

    SERVER_MODE = os.getenv("NIM_SERVER_MODE", "search").lower()
    print(f"Server mode: {SERVER_MODE}")

    if SERVER_MODE == "admin":
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
        print("Registered admin tools")
    elif SERVER_MODE == "manage":
        mcp.add_tool(index_file_by_path)
        mcp.add_tool(index_directory)
        print("Registered manage tools")
    else:
        print("Registered search tools: search_code")

    print()
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
