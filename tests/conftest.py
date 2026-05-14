import pytest


@pytest.fixture(autouse=True)
def setup_env_vars(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test_api_key")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./test_chroma_db")
    monkeypatch.setenv("NIM_SERVER_MODE", "admin")


@pytest.fixture
def mock_chroma_db(tmp_path):
    return str(tmp_path / "chroma_db")
