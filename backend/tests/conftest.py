import os
import tempfile

# Point the data dir at a throwaway temp dir before any app module imports it,
# so tests never read/write the real /app/data volume.
os.environ.setdefault("CHATDOC_DATA_DIR", tempfile.mkdtemp(prefix="chatdoc-test-"))

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from fake_providers.fake_qdrant import FakeQdrant


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        qdrant_host="localhost",
        qdrant_port=6333,
        chunk_size=128,
        chunk_overlap=16,
        top_k_dense=5,
        top_k_sparse=5,
        top_k_final=3,
    )


@pytest.fixture
def fake_qdrant() -> FakeQdrant:
    return FakeQdrant()


@pytest.fixture
def client(fake_qdrant) -> TestClient:
    from app.core import deps

    app = create_app()
    app.dependency_overrides[deps.get_qdrant] = lambda: fake_qdrant
    return TestClient(app)
