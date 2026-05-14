import os
import sys

import pytest

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from file_utils import (
    detect_language_from_extension,
    generate_document_id,
    normalize_file_content,
)


def test_generate_document_id():
    # Without filepath
    id1 = generate_document_id("some code")
    id2 = generate_document_id("some code")
    assert id1 == id2

    # With filepath
    id3 = generate_document_id("some code", filepath="/path/to/file.py")
    assert id1 != id3

    # Different content
    id4 = generate_document_id("other code")
    assert id1 != id4


def test_detect_language_from_extension():
    assert detect_language_from_extension("test.py") == "python"
    assert detect_language_from_extension("test.js") == "javascript"
    assert detect_language_from_extension("test.tsx") == "typescript"
    assert detect_language_from_extension("test.unknown") == "unknown"
    assert detect_language_from_extension("no_extension") == "unknown"


@pytest.mark.asyncio
async def test_normalize_file_content(tmp_path):
    test_file = tmp_path / "test.py"
    test_content = "print('Hello, world!')"

    # Test UTF-8
    test_file.write_text(test_content, encoding="utf-8")
    content = await normalize_file_content(str(test_file))
    assert content == test_content

    # Test alternative encoding (e.g. latin-1)
    test_file.write_bytes(test_content.encode("latin-1"))
    content = await normalize_file_content(str(test_file))
    assert content == test_content

    # Test binary file (will fall back to replace errors)
    test_file.write_bytes(b"\x80\x81\x82")
    content = await normalize_file_content(str(test_file))
    assert type(content) is str
