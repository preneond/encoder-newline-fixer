"""Offline tests for Hub publishing — the model card builder needs no network."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def publish_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "publish_model", REPO / "scripts" / "publish_model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_card_carries_metrics_and_serving_snippet(publish_module: ModuleType) -> None:
    config = {
        "model_name": "distilroberta-base",
        "max_words": 180,
        "val_metrics": {"accuracy": 0.9795, "macro_f1_structural": 0.7835, "f1_JOIN": 0.9361},
    }
    card = publish_module.build_model_card("someone/newlinefix-encoder", config)
    assert card.startswith("---\n")  # front matter first, so the Hub parses the tags
    assert "base_model: distilroberta-base" in card
    assert "| macro_f1_structural | 0.7835 |" in card
    assert 'EncoderGapPredictor.load("someone/newlinefix-encoder")' in card
    # Non-float entries (like max_words) must not leak into the metric table.
    assert "| max_words" not in card


def test_model_card_defaults_without_metrics(publish_module: ModuleType) -> None:
    card = publish_module.build_model_card("someone/repo", {})
    assert "base_model: distilroberta-base" in card
    assert "| metric | value |" in card
