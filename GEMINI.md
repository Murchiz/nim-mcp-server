# Project Overview: NVIDIA NIM MCP Server

This project is an MCP (Model Context Protocol) server that provides semantic code search capabilities. It uses NVIDIA NIM APIs for code embedding and reranking, and ChromaDB for persistent vector storage. The server also supports AST-based semantic chunking using `tree-sitter` for more accurate and structured context retrieval.

## Architecture & Technologies

- **MCP Framework**: Uses `FastMCP` from the `mcp` package to expose tools for indexing and searching code.
- **Embeddings & Reranking**: Integrates with NVIDIA NIM APIs (`nv-embedcode-7b-v1` for embeddings, `llama-nemotron-rerank-1b-v2` for reranking).
- **Vector Database**: `chromadb` is used for persistent local storage, typically saving to `./chroma_db`.
- **Semantic Chunking**: Uses `tree-sitter` (with language-specific parsers) to parse code into an Abstract Syntax Tree (AST) and split it by structural boundaries (functions, classes, interfaces, etc.) rather than arbitrary token limits.

## Development & Setup

- **Environment**: Python (configured for >=3.14 in `pyproject.toml`).
- **Dependency Management**: Uses `uv`.
  - Install dependencies: `uv sync`
- **Running the Server**: `uv run --directory . src/server.py`
- **Environment Variables**:
  - `NVIDIA_API_KEY`: (Required) For authenticating with NVIDIA NIM APIs.
  - `CHROMA_PERSIST_DIR`: (Optional) Defaults to `./chroma_db`.
  - `NIM_SERVER_MODE`: (Optional) Sets server tool availability mode (`search`, `manage`, or `admin`). Defaults to `search`.

## Directory Structure & Key Files

- `src/server.py`: The main entry point. It defines the `FastMCP` server, ChromaDB client management, NVIDIA API communication, and registers tools like `search_code`, `index_file_by_path`, and `index_directory`.
- `src/chunker.py`: Implements the AST parsing and chunking logic. It defines the mapping of file extensions to `tree-sitter` parsers, the logic to extract semantic nodes (`chunk_code_by_ast`), and context-injection formatting (`format_for_nvidia_nim`).
- `pyproject.toml`: Defines project metadata and lists required dependencies (including various `tree-sitter` language bindings).
- `README.md`: Contains high-level project documentation, usage examples, and IDE configuration instructions for MCP clients.

## Development Conventions

- **Asynchronous Programming**: The project uses `asyncio` for non-blocking operations. Ensure any synchronous I/O operations (like file reads, ChromaDB queries, or disk writes) are offloaded using `asyncio.to_thread()` to prevent blocking the MCP event loop.
- **API Concurrency**: External API calls use an asynchronous HTTP client (`httpx.AsyncClient`) with connection pooling and an `asyncio.Semaphore` to manage rate limiting.
- **Type Hinting**: Strict type hinting is enforced throughout the codebase (`typing.Dict`, `typing.List`, `typing.Optional`, `typing.Any`).
- **Error Handling pattern**: MCP tools generally catch internal exceptions and return a consistent dictionary structure, typically containing `"success": False` and an `"error"` message, rather than letting the server crash.
- **Tool Design**: Tools are designed to be state-aware for a specific workspace (using `db_path`), allowing isolated project indexes.