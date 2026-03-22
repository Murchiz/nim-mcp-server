#!/usr/bin/env python3
"""
AST-based Code Chunking Module for NVIDIA NIM Code Retrieval

This module provides semantic code chunking using tree-sitter AST parsers.
It extracts meaningful code blocks (functions, classes, etc.) for embedding.
"""

import os
from typing import Any, Dict, List, Optional

# Tree-sitter imports
try:
    import tree_sitter_c as tsc
    import tree_sitter_c_sharp as tscsharp
    import tree_sitter_cpp as tscpp
    import tree_sitter_go as tsgo
    import tree_sitter_java as tsjava
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_python as tspython
    import tree_sitter_rust as tsrust
    import tree_sitter_typescript as tstypescript
    import tree_sitter_zig as tszig
    from tree_sitter import Language, Parser
except ImportError as e:
    raise ImportError(
        f"Missing tree-sitter dependencies. Please install: {e}\n"
        "pip install tree-sitter>=0.23 tree-sitter-python tree-sitter-javascript "
        "tree-sitter-typescript tree-sitter-java tree-sitter-c tree-sitter-cpp "
        "tree-sitter-c-sharp tree-sitter-rust tree-sitter-zig tree-sitter-go"
    )

# =============================================================================
# AST Node Type Mapping (EXACT configuration as specified)
# =============================================================================
EXTENSION_TO_AST_NODES: Dict[str, List[str]] = {
    # Python
    ".py": ["function_definition", "class_definition"],
    # JavaScript
    ".js": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    ],
    ".jsx": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
    ],
    # TypeScript
    ".ts": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "arrow_function",
    ],
    ".tsx": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "arrow_function",
    ],
    # Java
    ".java": ["class_declaration", "method_declaration", "interface_declaration"],
    # C
    ".c": ["function_definition", "struct_specifier"],
    ".h": ["function_definition", "struct_specifier"],
    # C++
    ".cpp": ["function_definition", "class_specifier", "struct_specifier"],
    ".hpp": ["function_definition", "class_specifier", "struct_specifier"],
    ".cc": ["function_definition", "class_specifier", "struct_specifier"],
    ".cxx": ["function_definition", "class_specifier", "struct_specifier"],
    # C#
    ".cs": [
        "method_declaration",
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "namespace_declaration",
    ],
    # Rust
    ".rs": ["function_item", "impl_item", "struct_item", "trait_item"],
    # Zig
    ".zig": ["FunctionDecl", "ContainerDecl", "TestDecl"],
    # Go
    ".go": ["function_declaration", "method_declaration", "type_declaration"],
}

# =============================================================================
# Language Parser Configuration
# =============================================================================
# Language cache to avoid re-creating Language objects
_LANGUAGE_CACHE: Dict[str, Any] = {}


def _get_language(extension: str) -> Optional[Any]:
    """
    Get the tree-sitter Language object for a given file extension.

    Args:
        extension: File extension (e.g., '.py', '.js')

    Returns:
        Language object or None if not supported
    """
    if extension in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[extension]

    language_map = {
        ".py": tspython,
        ".js": tsjavascript,
        ".jsx": tsjavascript,
        ".ts": tstypescript,
        ".tsx": tstypescript,
        ".java": tsjava,
        ".c": tsc,
        ".h": tsc,
        ".cpp": tscpp,
        ".hpp": tscpp,
        ".cc": tscpp,
        ".cxx": tscpp,
        ".cs": tscsharp,
        ".rs": tsrust,
        ".zig": tszig,
        ".go": tsgo,
    }
    lang_module = language_map.get(extension)
    if lang_module is None:
        return None

    # Handle TypeScript special cases
    if extension == ".ts":
        language = Language(tstypescript.language_typescript())
    elif extension == ".tsx":
        language = Language(tstypescript.language_tsx())
    else:
        language = Language(lang_module.language())

    _LANGUAGE_CACHE[extension] = language
    return language


def _get_parser(language: Any) -> Parser:
    """
    Create a parser for the given language.

    Args:
        language: Tree-sitter Language object

    Returns:
        Configured Parser instance
    """
    return Parser(language)


# =============================================================================
# AST Traversal and Chunk Extraction
# =============================================================================
def _extract_node_text(source_code: str, node) -> str:
    """
    Extract the text content of a tree-sitter node from source code.

    Args:
        source_code: The full source code as a string
        node: Tree-sitter node

    Returns:
        The text content of the node
    """
    start_byte = node.start_byte
    end_byte = node.end_byte
    return source_code[start_byte:end_byte]


def _traverse_and_extract(
    source_code: str, node, target_types: List[str], chunks: List[Dict]
) -> None:
    """
    Recursively traverse the AST and extract nodes matching target types.

    IMPORTANT: If a parent node is extracted, we do NOT traverse into its children
    to avoid overlapping/duplicate chunks.

    Args:
        source_code: The full source code as a string
        node: Current tree-sitter node
        target_types: List of AST node types to extract
        chunks: List to append extracted chunks to
    """
    # Check if this node matches one of our target types
    if node.type in target_types:
        # Extract this node
        text = _extract_node_text(source_code, node)

        # Get line numbers (0-indexed)
        start_line = node.start_point[0]
        end_line = node.end_point[0]

        chunk = {
            "text": text,
            "type": node.type,
            "start_line": start_line,
            "end_line": end_line,
        }
        chunks.append(chunk)

        # Do NOT traverse children - we extracted the parent completely
        return

    # Traverse children if we didn't extract this node
    for child in node.children:
        _traverse_and_extract(source_code, child, target_types, chunks)


def chunk_code_by_ast(file_path: str, source_code: str) -> List[Dict]:
    """
    Parse source code using tree-sitter and extract semantic chunks based on AST nodes.

    This function:
    1. Detects the language from the file extension
    2. Loads the appropriate tree-sitter parser
    3. Parses the code into an AST
    4. Extracts nodes matching the target types for that language
    5. Returns non-overlapping chunks (parent nodes are extracted whole, children are skipped)

    Args:
        file_path: Path to the source file (used for extension detection)
        source_code: The source code content as a string

    Returns:
        List of dictionaries with keys:
        - text: The code snippet
        - type: The AST node type
        - start_line: Starting line number (0-indexed)
        - end_line: Ending line number (0-indexed)

    Note:
        If the file extension is not supported, returns the entire file as a single chunk.
    """
    # Get file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    # Check if extension is supported
    if extension not in EXTENSION_TO_AST_NODES:
        # Fallback: return entire file as single chunk
        lines = source_code.splitlines()
        return [
            {
                "text": source_code,
                "type": "file",
                "start_line": 0,
                "end_line": len(lines) - 1 if lines else 0,
            }
        ]

    # Get target AST node types for this extension
    target_types = EXTENSION_TO_AST_NODES[extension]

    # Get the language
    language = _get_language(extension)
    if language is None:
        # Fallback if language not available
        lines = source_code.splitlines()
        return [
            {
                "text": source_code,
                "type": "file",
                "start_line": 0,
                "end_line": len(lines) - 1 if lines else 0,
            }
        ]

    # Create parser and parse the code
    try:
        parser = _get_parser(language)
        tree = parser.parse(bytes(source_code, "utf-8"))
        root_node = tree.root_node
    except Exception:
        # If parsing fails, fallback to returning entire file
        lines = source_code.splitlines()
        return [
            {
                "text": source_code,
                "type": "file",
                "start_line": 0,
                "end_line": len(lines) - 1 if lines else 0,
            }
        ]

    # Traverse AST and extract chunks
    chunks: List[Dict] = []
    _traverse_and_extract(source_code, root_node, target_types, chunks)

    # If no chunks were extracted, return the entire file
    if not chunks:
        lines = source_code.splitlines()
        return [
            {
                "text": source_code,
                "type": "file",
                "start_line": 0,
                "end_line": len(lines) - 1 if lines else 0,
            }
        ]

    # Sort chunks by start_line
    chunks.sort(key=lambda x: x["start_line"])

    return chunks


# =============================================================================
# Helper function to split large chunks
# =============================================================================
def _split_large_chunk(
    chunk: Dict[str, Any], max_chars: int = 3500, min_chunk_chars: int = 500
) -> List[Dict[str, Any]]:
    """
    Split a large chunk into smaller pieces if it exceeds max_chars.

    This function splits chunks that are too large for embedding API limits.
    It tries to split at natural boundaries (blank lines, newlines) to preserve
    code readability.

    Args:
        chunk: Dictionary with 'text', 'type', 'start_line', 'end_line'
        max_chars: Maximum characters per chunk (default 3500 for safe margin)
        min_chunk_chars: Minimum characters per sub-chunk (avoid tiny chunks)

    Returns:
        List of chunk dictionaries, split if necessary
    """
    text = chunk["text"]

    # If chunk is small enough, return as-is
    if len(text) <= max_chars:
        return [chunk]

    # Split into smaller pieces
    sub_chunks = []
    lines = text.split("\n")
    current_text = ""
    current_start_line = chunk["start_line"]

    for i, line in enumerate(lines):
        test_text = current_text + ("\n" if current_text else "") + line

        # Check if adding this line would exceed the limit
        if len(test_text) > max_chars and current_text:
            # Save current chunk if it's big enough
            if len(current_text) >= min_chunk_chars:
                sub_chunks.append(
                    {
                        "text": current_text,
                        "type": chunk["type"],
                        "start_line": current_start_line,
                        "end_line": current_start_line + current_text.count("\n"),
                    }
                )
            current_text = line
            current_start_line = chunk["start_line"] + i
        else:
            # Current chunk too small, just add the line anyway
            current_text = test_text

    # Don't forget the last chunk
    if current_text and len(current_text) >= min_chunk_chars:
        sub_chunks.append(
            {
                "text": current_text,
                "type": chunk["type"],
                "start_line": current_start_line,
                "end_line": chunk["end_line"],
            }
        )
    elif current_text and sub_chunks:
        # Append remaining text to last chunk if possible
        last_chunk = sub_chunks[-1]
        if len(last_chunk["text"]) + len(current_text) <= int(max_chars * 1.5):
            sub_chunks[-1]["text"] += "\n" + current_text
            sub_chunks[-1]["end_line"] = chunk["end_line"]
        else:
            # Create a new small chunk anyway
            sub_chunks.append(
                {
                    "text": current_text,
                    "type": chunk["type"],
                    "start_line": current_start_line,
                    "end_line": chunk["end_line"],
                }
            )

    return sub_chunks if sub_chunks else [chunk]


# =============================================================================
# Contextual Injection for NVIDIA NIM Embeddings
# =============================================================================
def format_for_nvidia_nim(file_path: str, raw_chunks: List[Dict]) -> List[Dict]:
    """
    Format raw chunks with contextual metadata for NVIDIA NIM embedding.

    This function prepends context to each chunk's text field as comments,
    providing metadata about the file path, node type, and line numbers.

    Format:
        // File: {file_path}
        // Type: {node_type} | Lines: {start}-{end}
        [ACTUAL CHUNK CODE]

    Args:
        file_path: Path to the source file
        raw_chunks: List of raw chunk dictionaries from chunk_code_by_ast

    Returns:
        List of updated dictionaries with contextual headers prepended to text
    """
    formatted_chunks = []
    for chunk in raw_chunks:
        # Extract chunk metadata
        original_text = chunk["text"]
        node_type = chunk["type"]
        start_line = chunk["start_line"]
        end_line = chunk["end_line"]

        # Build contextual header (using // style comments for universal compatibility)
        # This works well for C-style languages and is readable for others
        context_lines = [
            f"// File: {file_path}",
            f"// Type: {node_type} | Lines: {start_line}-{end_line}",
            "",
        ]
        context_header = "\n".join(context_lines)

        # Prepend context to the chunk text
        formatted_text = context_header + original_text

        # Create updated chunk dictionary
        formatted_chunk = {
            "text": formatted_text,
            "type": node_type,
            "start_line": start_line,
            "end_line": end_line,
            "original_text": original_text,  # Keep original for reference
        }

        # Check if chunk exceeds maximum size and split if necessary
        # Using 3500 chars as safe limit (accounts for tokenization overhead)
        MAX_CHUNK_CHARS = 3500
        if len(formatted_text) > MAX_CHUNK_CHARS:
            # Split the original chunk (without context header)
            split_chunks = _split_large_chunk(
                chunk, max_chars=MAX_CHUNK_CHARS - len(context_header)
            )

            # Add context header to each split piece
            for split_chunk in split_chunks:
                split_context_lines = [
                    f"// File: {file_path}",
                    f"// Type: {node_type} | Lines: {split_chunk['start_line']}-{split_chunk['end_line']} (split)",
                    "",
                ]
                split_context_header = "\n".join(split_context_lines)
                split_formatted_text = split_context_header + split_chunk["text"]

                formatted_split_chunk = {
                    "text": split_formatted_text,
                    "type": node_type,
                    "start_line": split_chunk["start_line"],
                    "end_line": split_chunk["end_line"],
                    "original_text": split_chunk["text"],
                    "is_split": True,
                }
                formatted_chunks.append(formatted_split_chunk)
        else:
            formatted_chunks.append(formatted_chunk)

    return formatted_chunks


# =============================================================================
# Utility Functions
# =============================================================================
def get_supported_extensions() -> List[str]:
    """
    Get list of all supported file extensions for AST chunking.

    Returns:
        List of supported file extensions (e.g., ['.py', '.js', '.ts'])
    """
    return list(EXTENSION_TO_AST_NODES.keys())


def is_extension_supported(file_path: str) -> bool:
    """
    Check if a file extension is supported for AST chunking.

    Args:
        file_path: Path to the file

    Returns:
        True if the extension is supported, False otherwise
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower() in EXTENSION_TO_AST_NODES


def get_node_types_for_extension(extension: str) -> Optional[List[str]]:
    """Get the target AST node types for a given file extension.

    Args:
        extension: File extension (e.g., '.py')

    Returns:
        List of node types or None if extension not supported
    """
    return EXTENSION_TO_AST_NODES.get(extension.lower())


# =============================================================================
# Test/Demo
# =============================================================================
if __name__ == "__main__":
    # Simple test/demo
    test_python_code = """
def hello_world():
    \"\"\"A simple greeting function.\"\"\"
    print("Hello, World!")
    return True

class Calculator:
    \"\"\"A simple calculator class.\"\"\"
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

def main():
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
"""

    print("=" * 60)
    print("Testing Python AST Chunking")
    print("=" * 60)
    chunks = chunk_code_by_ast("test.py", test_python_code)
    print(f"\nExtracted {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(
            f"Chunk {i}: {chunk['type']} (lines {chunk['start_line']}-{chunk['end_line']})"
        )
        print("-" * 40)
        print(
            chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
        )
        print()

    print("=" * 60)
    print("Testing Contextual Formatting")
    print("=" * 60)
    formatted = format_for_nvidia_nim("test.py", chunks)
    print("\nFormatted chunks:\n")
    for i, chunk in enumerate(formatted[:2], 1):  # Show first 2
        print(f"Chunk {i}:")
        print("-" * 40)
        print(
            chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
        )
        print()
