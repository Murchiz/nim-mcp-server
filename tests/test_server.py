import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from server import (
    create_collection,
    delete_collection,
    delete_document,
    get_ast_chunking_info,
    get_collection_stats,
    get_supported_languages,
    health_check,
    index_directory,
    index_file_by_path,
    list_collections,
    search_code,
)


@pytest.mark.asyncio
async def test_health_check(mocker):
    mocker.patch("server.get_embedding", return_value=[0.1])
    # mock collection count
    mock_coll = mocker.MagicMock()
    mock_coll.count.return_value = 0
    mocker.patch("server.get_or_create_collection", return_value=mock_coll)

    status = await health_check()
    assert status["success"] is True
    assert status["api_connected"] is True
    assert status["chroma_connected"] is True
    assert status["healthy"] is True


def test_get_supported_languages():
    res = get_supported_languages()
    assert res["success"] is True
    assert ".py" in res["file_extensions"]


def test_get_ast_chunking_info():
    res = get_ast_chunking_info()
    assert res["success"] is True
    assert ".py" in res["supported_extensions"]


@pytest.mark.asyncio
async def test_create_and_list_collections(mock_chroma_db):
    res = await create_collection("test_col", db_path=mock_chroma_db)
    assert res["success"] is True

    res_list = await list_collections(db_path=mock_chroma_db)
    assert res_list["success"] is True
    assert "test_col" in res_list["collections"]


@pytest.mark.asyncio
async def test_get_collection_stats(mock_chroma_db):
    await create_collection("test_col2", db_path=mock_chroma_db)
    res = await get_collection_stats(db_path=mock_chroma_db, collection="test_col2")
    assert res["success"] is True
    assert res["document_count"] == 0


@pytest.mark.asyncio
async def test_delete_collection(mock_chroma_db):
    await create_collection("test_col3", db_path=mock_chroma_db)
    res = await delete_collection("test_col3", db_path=mock_chroma_db)
    assert res["success"] is True


@pytest.mark.asyncio
async def test_index_file_by_path_code_content(mock_chroma_db, mocker):
    mocker.patch("server.get_embedding", return_value=[0.1] * 10)

    res = await index_file_by_path(
        filepath="/fake/file.py",
        db_path=mock_chroma_db,
        collection="test_idx",
        code_content="print('hello')",
        use_ast_chunking=False,
    )
    assert res["success"] is True
    assert "document_id" in res


@pytest.mark.asyncio
async def test_index_file_by_path_ast(mock_chroma_db, mocker):
    mocker.patch("server.get_embeddings_batch", return_value=[[0.1] * 10])

    res = await index_file_by_path(
        filepath="/fake/file.py",
        db_path=mock_chroma_db,
        collection="test_idx_ast",
        code_content="def hello(): pass",
        use_ast_chunking=True,
    )
    assert res["success"] is True
    assert len(res["document_ids"]) == 1


@pytest.mark.asyncio
async def test_index_file_by_path_metadata_only(mock_chroma_db, mocker):
    mocker.patch("server.get_embedding", return_value=[0.1] * 10)
    res = await index_file_by_path(
        filepath="/fake/file2.py",
        db_path=mock_chroma_db,
        collection="test_idx_meta",
        code_content="print('hello')",
        use_content_as_document=False,
    )
    assert res.get("success") is True, res.get("error")
    assert res["method"] == "metadata_only"


@pytest.mark.asyncio
async def test_delete_document(mock_chroma_db, mocker):
    mocker.patch("server.get_embedding", return_value=[0.1] * 10)
    res_idx = await index_file_by_path(
        filepath="/fake/file3.py",
        db_path=mock_chroma_db,
        collection="test_del",
        code_content="print('hello')",
        use_ast_chunking=False,
    )

    res_del = await delete_document(
        res_idx["document_id"], db_path=mock_chroma_db, collection="test_del"
    )
    assert res_del["success"] is True


@pytest.mark.asyncio
async def test_search_code(mock_chroma_db, mocker):
    # Index something first
    mocker.patch("server.get_embedding", return_value=[0.1] * 10)
    await index_file_by_path(
        filepath="/fake/file4.py",
        db_path=mock_chroma_db,
        collection="test_search",
        code_content="def search_me(): pass",
        use_ast_chunking=False,
    )

    # Mock rerank
    mocker.patch("server.rerank", return_value=[{"index": 0, "logit": 1.0}])

    res = await search_code("search", db_path=mock_chroma_db, collection="test_search")

    assert res["success"] is True
    assert res["count"] == 1
    assert "search_me" in res["results"][0]["code"]


@pytest.mark.asyncio
async def test_index_directory(mock_chroma_db, mocker, tmp_path):
    mocker.patch("server.get_embedding", return_value=[0.1] * 10)
    mocker.patch("server.get_embeddings_batch", return_value=[[0.1] * 10])

    d = tmp_path / "src_test"
    d.mkdir()
    f = d / "test.py"
    f.write_text("def test(): pass")

    res = await index_directory(
        directory_path=str(d), db_path=mock_chroma_db, collection="test_dir"
    )
    assert res["success"] is True
    assert res["indexed_files"] == 1
    assert res["total_files"] == 1


@pytest.mark.asyncio
async def test_search_code_no_results(mock_chroma_db, mocker):
    mocker.patch("server.get_embedding", return_value=[0.1] * 10)

    res = await search_code(
        "search", db_path=mock_chroma_db, collection="test_search_empty"
    )

    assert res["success"] is True
    assert res["count"] == 0
    assert res["results"] == []
