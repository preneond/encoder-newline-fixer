"""Self-supervised training-pair generation.

A clean document already carries its own labels: the true gap classes. Corruption
produces the *model input* view (a word sequence with newline structure destroyed
and occasional mid-word splits) while keeping labels aligned.

Two views are produced:
  * ``make_example`` — (input words, true gap labels): what models train/predict on.
    Models deliberately never see the input's own (unreliable) newline placement.
  * ``render_corrupted`` — a plausible broken input *text* for those words, used for
    end-to-end demos and service-level tests, never for training.
"""

import random
from dataclasses import dataclass

from newlinefix.gaps import JOIN, NEWLINE, PARA, SPACE, GapText


@dataclass(frozen=True)
class CorruptionConfig:
    """Rates are chosen to mimic copy-paste/PDF-extraction damage.

    p_word_split stays low so the JOIN prior is conservative: merging two words
    wrongly is a worse failure than leaving a split word unrepaired.
    """

    p_word_split: float = 0.02
    min_split_len: int = 4
    # Rendering-only knobs (broken input text for e2e evaluation):
    p_spurious_newline: float = 0.05  # a SPACE gap rendered as "\n" (hard-wrap damage)
    p_keep_true_newline: float = 0.15  # a true NEWLINE/PARA that survives rendering
    p_split_rendered_as_newline: float = 0.8  # JOIN rendered "\n" vs " "


def make_example(
    clean: GapText, rng: random.Random, cfg: CorruptionConfig
) -> tuple[list[str], list[int]]:
    """Return (input_words, true_gap_labels) aligned to the corrupted word sequence.

    The only corruption that changes the word sequence is splitting a word in two
    (true gap JOIN); destroyed newlines don't alter words, so all other true gaps
    carry over unchanged.
    """
    words: list[str] = []
    labels: list[int] = []
    min_split_len = max(cfg.min_split_len, 2)
    for i, word in enumerate(clean.words):
        if len(word) >= min_split_len and rng.random() < cfg.p_word_split:
            cut = rng.randint(1, len(word) - 1)
            words.extend((word[:cut], word[cut:]))
            labels.append(JOIN)
        else:
            words.append(word)
        if i < len(clean.gaps):
            labels.append(clean.gaps[i])
    return words, labels


def render_corrupted(
    words: list[str], labels: list[int], rng: random.Random, cfg: CorruptionConfig
) -> str:
    """Render a broken input text whose word sequence is exactly ``words``.

    Every gap renders as non-empty whitespace, so
    ``text_to_gaps(render_corrupted(...)).words == words`` holds by construction.
    """
    parts = [words[0]] if words else []
    for label, word in zip(labels, words[1:], strict=True):
        if label == JOIN:
            sep = "\n" if rng.random() < cfg.p_split_rendered_as_newline else " "
        elif label in (NEWLINE, PARA) and rng.random() < cfg.p_keep_true_newline:
            sep = "\n" if label == NEWLINE else "\n\n"
        elif label == SPACE and rng.random() < cfg.p_spurious_newline:
            sep = "\n"
        else:
            sep = " "
        parts.append(sep)
        parts.append(word)
    return "".join(parts)
