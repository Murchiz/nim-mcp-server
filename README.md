# NVIDIA NIM MCP Server

An MCP (Model Context Protocol) server for semantic code search using NVIDIA NIM embeddings and reranking, with ChromaDB for persistent vector storage.

## Features

- **Semantic code search** with natural language queries
- **Automatic reranking** for improved result relevance
- **AST-based chunking** of functions, classes, and other structural units
- **Persistent ChromaDB storage** with per-workspace isolation
- **Batch indexing** for directories and single files

## Prerequisites

- Python 3.14+
- An NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com/explore/discover)

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Set your API key:

```bash
# PowerShell
$env:NVIDIA_API_KEY = "your-api-key-here"

# Linux / macOS
export NVIDIA_API_KEY="your-api-key-here"
```

## IDE Configuration

Add the server to your MCP client settings (e.g., in Zed, Claude Desktop, or Cline):

```json
{
  "mcpServers": {
    "nim-code-search": {
      "command": "/absolute/path/to/nim-mcp-server/.venv/Scripts/python.exe",
      "args": ["/absolute/path/to/nim-mcp-server/src/server.py"],
      "env": {
        "NVIDIA_API_KEY": "your-api-key-here",
        /// Optional, but recommended
        "NIM_SERVER_MODE":"manage",
        "CHROMA_PERSIST_DIR":"./chroma_db"
      }
    }
  }
}
```

> **Note**: `CHROMA_PERSIST_DIR` defaults to `./chroma_db` (relative to the working directory). When you use this MCP server in a project, the database will be created in that project's folder, keeping each project's indexed code separate. You can override this by adding `CHROMA_PERSIST_DIR` to the `env` section.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | *(required)* | Your NVIDIA API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `NIM_SERVER_MODE` | `search` | Server tool availability (`search`, `manage`, or `admin`) |

## Server Modes

`NIM_SERVER_MODE` controls which tools are exposed by the server, letting you tailor the server to a specific workflow:

| Mode | Available Tools |
|------|-----------------|
| `search` *(default)* | `search_code` — query already-indexed code |
| `manage` | `search_code`, `index_file_by_path`, `index_directory` — search **and** index new code |
| `admin` | All tools — full access including collection management, health checks, and language info |

## Tools

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search with automatic reranking |
| `index_file_by_path` | Index a single file by path |
| `index_directory` | Recursively index all supported code files |
| `delete_document` | Delete a document by ID |
| `delete_collection` | Delete an entire collection |
| `list_collections` | List all collections |
| `create_collection` | Create a new collection |
| `get_collection_stats` | Get document count and collection info |
| `get_supported_languages` | List supported languages and extensions |
| `get_ast_chunking_info` | AST chunking configuration |
| `health_check` | Check server and API status |

## How It Works

**Indexing**: Source code is parsed by `tree-sitter`, split into semantic chunks, embedded via the NVIDIA NIM API, and stored in ChromaDB with metadata.

**Search**: A natural language query is embedded, matched against the ChromaDB index, and the top results are reranked by the NVIDIA NIM reranker for maximum relevance.

## License

MIT
