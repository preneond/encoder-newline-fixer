"""Tests for the encoder gap predictor and its subtoken-alignment helpers.

Alignment logic is tested offline; model-loading tests use a tiny hub checkpoint
and skip gracefully when the hub is unreachable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from newlinefix.gaps import NEWLINE, NUM_GAP_CLASSES, PARA, SPACE
from newlinefix.models.encoder import EncoderGapPredictor, last_subtoken_positions

REPO = Path(__file__).resolve().parent.parent
TINY_MODEL = "sshleifer/tiny-distilroberta-base"


def load_train_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "train_encoder", REPO / "scripts" / "train_encoder.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLastSubtokenPositions:
    def test_specials_and_multi_subtoken_words(self) -> None:
        # [CLS] w0 w0 w1 w2 w2 w2 [SEP]
        word_ids: list[int | None] = [None, 0, 0, 1, 2, 2, 2, None]
        assert last_subtoken_positions(word_ids, 3) == [2, 3, 6]

    def test_truncated_words_map_to_none(self) -> None:
        # Words 2 and 3 were truncated away by the max-length limit.
        word_ids: list[int | None] = [None, 0, 1, 1]
        assert last_subtoken_positions(word_ids, 4) == [1, 3, None, None]

    def test_no_words(self) -> None:
        assert last_subtoken_positions([None, None], 0) == []


class TestGapLabelTensor:
    def test_labels_land_on_last_subtoken_of_left_word(self) -> None:
        mod = load_train_module()
        word_id_rows: list[list[int | None]] = [
            [None, 0, 0, 1, None],  # w0 split into two subtokens
            [None, 0, 1, None, None],  # shorter row, padded
        ]
        windows = [(["ab", "cd"], [NEWLINE]), (["x", "y"], [PARA])]
        labels = mod.gap_label_tensor(word_id_rows, windows, 5)
        assert labels.shape == (2, 5)
        assert labels[0].tolist() == [-100, -100, NEWLINE, -100, -100]
        assert labels[1].tolist() == [-100, PARA, -100, -100, -100]

    def test_truncated_left_word_gets_no_label(self) -> None:
        mod = load_train_module()
        # 3 words but word 2 truncated: gap 1 (left word 1) still labeled,
        # a hypothetical gap with truncated left word is skipped.
        word_id_rows: list[list[int | None]] = [[None, 0, 1, None]]
        windows = [(["a", "b", "c"], [SPACE, NEWLINE])]
        labels = mod.gap_label_tensor(word_id_rows, windows, 4)
        assert labels[0].tolist() == [-100, SPACE, NEWLINE, -100]

        # Now word 1 and 2 truncated: only gap 0 labeled.
        word_id_rows = [[None, 0, 0, None]]
        labels = mod.gap_label_tensor(word_id_rows, windows, 4)
        assert labels[0].tolist() == [-100, -100, SPACE, -100]


@pytest.fixture(scope="module")
def tiny_predictor(tmp_path_factory: pytest.TempPathFactory) -> EncoderGapPredictor:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(
            TINY_MODEL, num_labels=NUM_GAP_CLASSES
        )
    except Exception as exc:
        pytest.skip(f"tiny hub model unavailable (offline?): {exc}")
    assert tokenizer is not None
    save_dir = tmp_path_factory.mktemp("tiny-encoder")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    return EncoderGapPredictor.load(save_dir, device="cpu")


class TestEncoderGapPredictor:
    def test_predict_window_count_and_range(self, tiny_predictor: EncoderGapPredictor) -> None:
        words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        labels = tiny_predictor.predict_window(words)
        assert len(labels) == len(words) - 1
        assert all(0 <= lab < NUM_GAP_CLASSES for lab in labels)

    def test_space_fallback_when_words_truncated(self, tiny_predictor: EncoderGapPredictor) -> None:
        # 180 long words blow well past 512 subtokens, so trailing words are
        # truncated away and their gaps must fall back to SPACE.
        words = ["antidisestablishmentarianism" * 2] * 180
        labels = tiny_predictor.predict_window(words)
        assert len(labels) == 179
        assert labels[-1] == SPACE

    def test_predict_windows_batched_matches_shapes(
        self, tiny_predictor: EncoderGapPredictor
    ) -> None:
        windows = [
            ["alpha", "beta", "gamma"],
            ["one", "two", "three", "four", "five", "six"],
        ]
        results = tiny_predictor.predict_windows(windows)
        assert [len(r) for r in results] == [2, 5]
