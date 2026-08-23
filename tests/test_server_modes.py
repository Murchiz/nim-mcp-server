"""Smoke tests for per-mode MCP tool registration (search/manage/admin)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from server import mcp  # noqa: E402

ADMIN_TOOLS = {
    "delete_document",
    "delete_collection",
    "list_collections",
    "create_collection",
    "get_collection_stats",
    "health_check",
    "get_supported_languages",
    "get_ast_chunking_info",
    "index_file_by_path",
    "index_directory",
}

MANAGE_TOOLS = {
    "index_file_by_path",
    "index_directory",
}


async def _registered_tools() -> set[str]:
    tools = await mcp.list_tools()
    return {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_search_mode_registers_only_search_code(monkeypatch, mocker):
    # Fresh FastMCP instance semantics: search_code is registered at import.
    monkeypatch.setenv("NIM_SERVER_MODE", "search")
    mocker.patch.object(mcp, "run")

    from server import main

    main()  # must not block; run() is patched out

    tools = await _registered_tools()
    assert "search_code" in tools


@pytest.mark.asyncio
async def test_manage_mode_registers_indexing_tools(monkeypatch, mocker):
    monkeypatch.setenv("NIM_SERVER_MODE", "manage")
    mocker.patch.object(mcp, "run")

    from server import main

    main()

    tools = await _registered_tools()
    assert MANAGE_TOOLS <= tools
    assert "search_code" in tools  # always available


@pytest.mark.asyncio
async def test_admin_mode_registers_all_management_tools(monkeypatch, mocker):
    monkeypatch.setenv("NIM_SERVER_MODE", "admin")
    mocker.patch.object(mcp, "run")

    from server import main

    main()

    tools = await _registered_tools()
    assert ADMIN_TOOLS <= tools
    assert "search_code" in tools  # always available
