import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chunker import (
    _split_large_chunk,
    chunk_code_by_ast,
    format_for_nvidia_nim,
    get_node_types_for_extension,
    get_supported_extensions,
    is_extension_supported,
)


def test_chunk_code_by_ast_python():
    code = "def hello():\n    print('hello')\nclass Test:\n    def method(self):\n        pass\n"
    chunks = chunk_code_by_ast("test.py", code)
    assert len(chunks) == 2


def test_chunk_code_by_ast_unsupported():
    chunks = chunk_code_by_ast("test.txt", "just some text")
    assert len(chunks) == 1


def test_chunk_code_by_ast_fallback():
    chunks = chunk_code_by_ast("test.py", "syntax error >>>")
    assert len(chunks) == 1


def test_split_large_chunk():
    large_text = "line1\nline2\nline3\n" * 200
    chunk = {
        "text": large_text,
        "type": "function_definition",
        "start_line": 0,
        "end_line": 200 * 4,
    }
    split_chunks = _split_large_chunk(chunk, max_chars=100, min_chunk_chars=10)
    assert len(split_chunks) > 1


def test_format_for_nvidia_nim():
    chunks = [
        {
            "text": "def foo(): pass",
            "type": "function_definition",
            "start_line": 1,
            "end_line": 1,
        }
    ]
    formatted = format_for_nvidia_nim("test.py", chunks)
    assert len(formatted) == 1


def test_get_supported_extensions():
    assert ".py" in get_supported_extensions()


def test_is_extension_supported():
    assert is_extension_supported("test.py") is True
    assert is_extension_supported("test.txt") is False


def test_get_node_types_for_extension():
    nodes = get_node_types_for_extension(".py")
    assert nodes is not None
    assert "function_definition" in nodes
    assert get_node_types_for_extension(".txt") is None


def test_chunk_code_by_ast_various_languages():
    # JS
    chunks = chunk_code_by_ast("test.js", "function test() {}")
    assert chunks[0]["type"] == "function_declaration"

    # TS
    chunks = chunk_code_by_ast("test.ts", "interface A {}")
    assert chunks[0]["type"] == "interface_declaration"

    # Java
    chunks = chunk_code_by_ast("test.java", "class Test {}")
    assert chunks[0]["type"] == "class_declaration"

    # C
    chunks = chunk_code_by_ast("test.c", "void test() {}")
    assert chunks[0]["type"] == "function_definition"

    # CPP
    chunks = chunk_code_by_ast("test.cpp", "class Test {};")
    assert chunks[0]["type"] == "class_specifier"

    # C#
    chunks = chunk_code_by_ast("test.cs", "class Test {}")
    assert chunks[0]["type"] == "class_declaration"

    # Rust
    chunks = chunk_code_by_ast("test.rs", "fn test() {}")
    assert chunks[0]["type"] == "function_item"

    # Go
    chunks = chunk_code_by_ast("test.go", "func test() {}")
    assert chunks[0]["type"] == "function_declaration"


def test_chunk_code_with_orphaned_nodes():
    code = """
class MyClass:
    a = 1
    def __init__(self):
        pass
    b = 2
"""
    chunks = chunk_code_by_ast("test.py", code)
    # the whole class should be extracted because it's in target types.
    assert len(chunks) == 1
    assert chunks[0]["type"] == "class_definition"


def test_chunk_code_large_node_falls_through():
    # If a node is > max_chars, it falls through to children.
    # Python target types: function_definition, class_definition
    code = "class MyClass:\n" + "    a = 1\n" * 100 + "    def foo(): pass\n"
    # Wait, chunk_code_by_ast has max_chars hardcoded to 3500 in _traverse_and_extract
    # Let's make it bigger than 3500
    code = (
        "class MyClass:\n"
        + "    a = 1" * 100
        + "\n" * 100
        + "    def foo():\n        pass\n" * 100
    )
    chunks = chunk_code_by_ast("test.py", code)
    assert len(chunks) > 1


def test_split_large_chunk_edge_cases():
    # exactly at limit
    chunk = {"text": "a" * 3500, "type": "func", "start_line": 0, "end_line": 1}
    assert len(_split_large_chunk(chunk)) == 1

    # small piece at end
    chunk = {"text": "a" * 3500 + "\nb", "type": "func", "start_line": 0, "end_line": 1}
    res = _split_large_chunk(chunk)
    assert len(res) == 1  # appended to last chunk

    chunk = {
        "text": "a" * 3500 + "\n" + "b" * 1000,
        "type": "func",
        "start_line": 0,
        "end_line": 2,
    }
    res = _split_large_chunk(chunk, max_chars=3500, min_chunk_chars=500)
    assert len(res) == 2


def test_format_for_nvidia_nim_split():
    chunks = [
        {
            "text": ("a" * 400 + "\n") * 10,
            "type": "func",
            "start_line": 1,
            "end_line": 10,
        }
    ]
    res = format_for_nvidia_nim("test.py", chunks)
    assert len(res) > 1
    assert res[0].get("is_split") is True
