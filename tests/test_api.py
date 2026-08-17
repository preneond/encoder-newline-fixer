"""Service-level tests for the HTTP API.

The real encoder is never loaded: a rule baseline is injected through FastAPI's
dependency overrides, so these tests exercise the request/response contract and
the fix_text glue, not the model.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from newlinefix.api import app, get_predictor
from newlinefix.models.baseline import RuleBaseline


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_predictor] = RuleBaseline
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fix_replaces_whitespace_only(client: TestClient) -> None:
    # The rule baseline puts a newline before a bullet and a space everywhere else;
    # the input's own (unreliable) newlines are discarded.
    response = client.post("/fix", json={"text": "three\nways: • In attention layers"})
    assert response.status_code == 200
    assert response.json() == {"text": "three ways:\n• In attention layers"}


def test_fix_empty_text(client: TestClient) -> None:
    response = client.post("/fix", json={"text": ""})
    assert response.status_code == 200
    assert response.json() == {"text": ""}


def test_fix_missing_field_is_validation_error(client: TestClient) -> None:
    assert client.post("/fix", json={}).status_code == 422
