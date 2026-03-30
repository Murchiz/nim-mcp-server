# NVIDIA NIM MCP Server

An MCP server for semantic code search using NVIDIA NIM embeddings and reranking, with ChromaDB for vector storage.

## Features

- Semantic code search with natural language queries
- Automatic reranking for better relevance
- AST-based chunking (functions, classes, etc.)
- Persistent ChromaDB storage
- Batch indexing and directory indexing

## Prerequisites

- Python 3.8+
- NVIDIA API Key from [build.nvidia.com](https://build.nvidia.com/explore/discover)

## Setup

```bash
uv sync
```

Set your API key:

```bash
# PowerShell
$env:NVIDIA_API_KEY = "your-api-key-here"
# Linux/macOS
export NVIDIA_API_KEY="your-api-key-here"
```

## IDE Configuration

```json
{
  "mcpServers": {
    "nim-code-search": {
      "command": "uv",
      "args": ["run", "src/server.py"],
      "env": {
        "NVIDIA_API_KEY": "your-api-key-here",
        "CHROMA_PERSIST_DIR": "./chroma_db"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | (required) | Your NVIDIA API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |

## Tools

| Tool | Description |
|------|-------------|
| `index_code` | Index a code snippet with AST chunking |
| `index_file_by_path` | Index a single file by path |
| `index_directory` | Index all code files in a directory |
| `batch_index_codes` | Index multiple snippets in batch |
| `search_code` | Semantic search with automatic reranking |
| `delete_document` | Delete a document by ID |
| `delete_collection` | Delete a collection and all its documents |
| `list_collections` | List all collections |
| `create_collection` | Create a new collection |
| `get_collection_stats` | Get collection statistics |
| `get_supported_languages` | List supported languages |
| `get_ast_chunking_info` | AST chunking configuration |
| `health_check` | Check server and API status |

## How It Works

**Indexing**: Code → NVIDIA NIM Embedding API → ChromaDB (with metadata)

**Search**: Query → NVIDIA NIM Embedding API → ChromaDB similarity search → NVIDIA NIM Reranking API → Results

## License

MIT
