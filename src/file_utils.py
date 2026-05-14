import asyncio
import hashlib
import os

MAX_FILE_SIZE = 1024 * 1024  # 1MB

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

EXTENSION_TO_LANGUAGE = {}
for language, extensions in SUPPORTED_EXTENSIONS.items():
    for ext in extensions:
        EXTENSION_TO_LANGUAGE[ext] = language

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


def generate_document_id(text: str, filepath: str = "") -> str:
    """Generate a unique ID for a document based on its content and filepath."""
    content_to_hash = f"{filepath}::{text}" if filepath else text
    return hashlib.sha256(content_to_hash.encode()).hexdigest()[:16]


def detect_language_from_extension(file_path: str) -> str:
    """Detect programming language from file extension."""
    _, ext = os.path.splitext(file_path)
    return EXTENSION_TO_LANGUAGE.get(ext, "unknown")


async def normalize_file_content(file_path: str) -> str:
    """Read file content with proper encoding detection."""

    def _read_file():
        encodings_to_try = ["utf-8", "ascii"]

        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeError:
                continue

        # Fallback to utf-8 with replace
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    return await asyncio.to_thread(_read_file)
